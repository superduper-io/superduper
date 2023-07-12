import collections
import uuid

from flask import Flask, jsonify

from superduperdb.core.artifact import InMemoryArtifacts
from superduperdb.datalayer.base.build import build_datalayer
from superduperdb import CFG

from flask import request, make_response

from superduperdb.core.documents import Document, ArtifactDocument
from superduperdb.core.serializable import Serializable


def make_endpoints(app, db):

    cache = collections.defaultdict(lambda: InMemoryArtifacts())

    @app.route('/select', methods=['POST'])
    def select():
        d = request.get_json()['query']
        query = Serializable.from_dict(d)
        output = list(db.execute(query))
        return make_response(Document.dumps_many(output))

    @app.route('/select_one', methods=['POST'])
    def select_one():
        d = request.get_json()
        query = Serializable.from_dict(d['query'])
        output = db.execute(query)
        return make_response(Document.dumps(output))

    @app.route('/insert', methods=['POST'])
    def insert():
        d = request.get_json()
        binary = cache[d['request_id']].cache[d['documents']]
        del cache[d['request_id']]
        documents = Document.loads_many(binary, encoders=db.encoders)
        insert = Serializable.from_dict(d['query'])
        insert.documents = documents
        db.execute(insert)
        return 'ok'

    @app.route('/delete', methods=['POST'])
    def delete():
        d = request.get_json()['query']
        query = Serializable.from_dict(d)
        db.execute(query)
        return 'ok'

    @app.route('/add', methods=['POST'])
    def add():
        d = request.get_json()
        ad = ArtifactDocument(d)
        ad.load_artifacts(artifact_store=cache[d['request_id']], cache={})
        object = Serializable.from_dict(d)
        db.add(object)
        return 'ok'

    @app.route('/artifacts/get/<request_id>/<file_id>', methods=['GET'])
    def get(request_id, file_id):
        bytes = cache[request_id].cache[file_id]
        del cache[request_id].cache[file_id]
        if not cache[request_id]:
            del cache[request_id]
        return make_response(bytes)

    @app.route('/artifacts/put/<request_id>/<file_id>', methods=['POST'])
    def put(request_id, file_id):
        cache[request_id].cache[file_id] = request.get_data()
        return 'ok'

    @app.route('/load', methods=['POST'])
    def load():
        d = request.get_json()
        request_id = d['request_id']
        m = db.load(variety=d['variety'], identifier=d['identifier'], version=d.get('version'))
        ad = ArtifactDocument(m.to_dict())
        ad.save_artifacts(artifact_store=cache[request_id], cache={})
        print(ad.content)
        return jsonify(ad.content)

    @app.route('/remove', methods=['POST'])
    def remove():
        d = request.get_json()
        db.remove(
            variety=d['variety'],
            identifier=d['identifier'],
            version=d.get('version'),
            force=True,
        )
        return 'ok'

    @app.route('/show', methods=['GET'])
    def show():
        return jsonify(db.show(**request.args))


def serve():
    app = Flask(__name__)
    db = build_datalayer()
    make_endpoints(app, db)
    app.run(CFG.server.host, CFG.server.port)


if __name__ == '__main__':
    serve()
