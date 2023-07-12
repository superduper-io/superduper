import collections
import uuid
import traceback

from flask import Flask, jsonify, make_response, request

from superduperdb.core.artifact import InMemoryArtifacts
from superduperdb.datalayer.base.build import build_datalayer
from superduperdb import CFG

from superduperdb.core.documents import Document, ArtifactDocument
from superduperdb.core.serializable import Serializable


def make_endpoints(app, db):
    cache = collections.defaultdict(lambda: InMemoryArtifacts())

    @app.route('/select', methods=['GET'])
    def select():
        d = request.get_json()
        query = Serializable.deserialize(d['query'])
        output = list(db.execute(query))  # TODO - support cursor with flask streaming
        file_id = str(uuid.uuid4())
        cache[d['request_id']].cache[file_id] = Document.dump_bson(output)
        return jsonify({'file_id': file_id})

    @app.route('/select_one', methods=['GET'])
    def select_one():
        d = request.get_json()
        query = Serializable.deserialize(d['query'])
        output = db.execute(query)
        file_id = str(uuid.uuid4())
        cache[d['request_id']].cache[file_id] = Document.dump_bson(output)
        return jsonify({'file_id': file_id})

    @app.route('/insert', methods=['POST'])
    def insert():
        d = request.get_json()
        binary = cache[d['request_id']].cache[d['documents']]
        del cache[d['request_id']]
        documents = Document.load_bsons(binary, encoders=db.encoders)
        query = Serializable.deserialize(d['query'])
        query.documents = documents
        db.execute(query)
        return jsonify({'msg': 'ok'})

    @app.route('/update', methods=['POST'])
    def update():
        d = request.get_json()
        binary = cache[d['request_id']].cache[d['update']]
        del cache[d['request_id']]
        update = Document.load_bson(binary, encoders=db.encoders)
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
        ad = ArtifactDocument(d)
        ad.load_artifacts(artifact_store=cache[d['request_id']], cache={})
        object = Serializable.deserialize(d)
        db.add(object)
        return jsonify({'msg': 'ok'})

    @app.route('/artifacts/get/<request_id>/<file_id>', methods=['GET'])
    def get(request_id, file_id):
        bytes = cache[request_id].cache[file_id]
        del cache[request_id].cache[file_id]
        if not cache[request_id]:
            del cache[request_id]
        return make_response(bytes)

    @app.route('/artifacts/put/<request_id>/<file_id>', methods=['PUT'])
    def put(request_id, file_id):
        cache[request_id].cache[file_id] = request.get_data()
        return jsonify({'msg': 'ok'})

    @app.route('/load', methods=['GET'])
    def load():
        d = request.get_json()
        request_id = d['request_id']
        version = d.get('version')
        if version:
            version = int(version)
        m = db.load(variety=d['variety'], identifier=d['identifier'], version=version)
        ad = ArtifactDocument(m.serialize())
        ad.save_artifacts(artifact_store=cache[request_id], cache={})
        print(ad.content)
        return jsonify(ad.content)

    @app.route('/remove', methods=['POST'])
    def remove():
        d = request.get_json()
        version = d.get('version')
        if version:
            version = int(version)
        db.remove(
            variety=d['variety'],
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


def serve():
    app = Flask(__name__)
    db = build_datalayer()
    make_endpoints(app, db)
    app.run(CFG.server.host, CFG.server.port)


if __name__ == '__main__':
    serve()
