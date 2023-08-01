import collections
import traceback
import uuid

from flask import Flask, jsonify, make_response, request

from superduperdb import CFG
from superduperdb.container.artifact_tree import (
    get_artifacts,
    load_artifacts_from_store,
    replace_artifacts_with_dict,
)
from superduperdb.container.document import load_bson, load_bsons
from superduperdb.container.serializable import Serializable
from superduperdb.db.base.build import build_datalayer
from superduperdb.misc.serialization import serializers


def make_endpoints(app, db):
    cache = collections.defaultdict(lambda: {})

    @app.route('/select', methods=['GET'])
    def select():
        d = request.get_json()
        query = Serializable.deserialize(d['query'])
        output = list(db.execute(query))  # TODO - support cursor with flask streaming
        file_id = str(uuid.uuid4())
        cache[d['request_id']][file_id] = output.dump_bson()
        return jsonify({'file_id': file_id})

    @app.route('/select_one', methods=['GET'])
    def select_one():
        d = request.get_json()
        query = Serializable.deserialize(d['query'])
        output = db.execute(query)
        file_id = str(uuid.uuid4())
        cache[d['request_id']][file_id] = output.dump_bson()
        return jsonify({'file_id': file_id})

    @app.route('/insert', methods=['POST'])
    def insert():
        d = request.get_json()
        binary = cache[d['request_id']][d['documents']]
        del cache[d['request_id']]
        documents = load_bsons(binary, encoders=db.encoders)
        query = Serializable.deserialize(d['query'])
        query.documents = documents
        db.execute(query)
        return jsonify({'msg': 'ok'})

    @app.route('/update', methods=['POST'])
    def update():
        d = request.get_json()
        binary = cache[d['request_id']][d['update']]
        del cache[d['request_id']]
        update = load_bson(binary, encoders=db.encoders)
        query = Serializable.deserialize(d['query'])
        query.update = update
        db.execute(query)
        return jsonify({'msg': 'ok'})

    @app.route('/delete', methods=['POST'])
    def delete():
        d = request.get_json()['query']
        query = Serializable.deserialize(d)
        db.execute(query)
        return jsonify({'msg': 'ok'})

    @app.route('/add', methods=['POST'])
    def add():
        d = request.get_json()
        artifact_cache = cache[d['request_id']]
        for a in artifact_cache:
            artifact_cache[a] = serializers[d['serializers'][a]].decode(
                artifact_cache[a]
            )
        d = load_artifacts_from_store(d, cache=artifact_cache)
        object = Serializable.deserialize(d['component'])
        db.add(object)
        del cache[d['request_id']]
        return jsonify({'msg': 'ok'})

    @app.route('/artifacts/get/<request_id>/<file_id>', methods=['GET'])
    def get(request_id, file_id):
        bytes = cache[request_id][file_id]
        del cache[request_id][file_id]
        if not cache[request_id]:
            del cache[request_id]
        return make_response(bytes)

    @app.route('/artifacts/put/<request_id>/<file_id>', methods=['PUT'])
    def put(request_id, file_id):
        cache[request_id][file_id] = request.get_data()
        return jsonify({'msg': 'ok'})

    @app.route('/load', methods=['GET'])
    def load():
        d = request.get_json()
        request_id = d['request_id']
        version = d.get('version')
        if version:
            version = int(version)
        m = db.load(type_id=d['type_id'], identifier=d['identifier'], version=version)
        to_send = m.serialize()
        artifacts = list(get_artifacts(to_send))
        lookup = {a: str(uuid.uuid4()) for a in artifacts}
        s_lookup = {lookup[a]: a.serializer for a in artifacts}
        to_send = replace_artifacts_with_dict(to_send, lookup)
        for a in artifacts:
            cache[request_id][lookup[a]] = serializers[s_lookup[lookup[a]]].encode(
                a.artifact
            )
        return jsonify(to_send)

    @app.route('/remove', methods=['POST'])
    def remove():
        d = request.get_json()
        version = d.get('version')
        if version:
            version = int(version)

        db.remove(
            type_id=d['type_id'],
            identifier=d['identifier'],
            version=version,
            force=True,
        )
        return jsonify({'msg': 'ok'})

    @app.route('/show', methods=['GET'])
    def show():
        return jsonify(db.show(**request.get_json()))

    @app.errorhandler(500)
    def handle_500(error):
        error_traceback = traceback.format_exc()
        response = {
            'traceback': error_traceback,
            'error': str(error),
            'type': str(type(error)),
        }
        return jsonify(response), 500


def serve(db):
    app = Flask(__name__)
    make_endpoints(app, db)
    return app


def main():
    db = build_datalayer()
    app = serve(db)
    app.run(CFG.server.host, CFG.server.port)


if __name__ == '__main__':
    main()
