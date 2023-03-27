from superduperdb.mongodb.client import SuperDuperClient
from bson import BSON, ObjectId
from flask import request, Flask, jsonify, make_response

# https://flask.palletsprojects.com/en/2.1.x/patterns/streaming/ streaming for the find endpoint

from superduperdb import cf

app = Flask(__name__)

client = SuperDuperClient(**cf['mongodb'])
collections = {}


@app.route('/unset_hash_set', methods=['PUT'])
def unset_hash_set():
    data = request.get_json()
    database = data['database']
    collection = data['collection']
    if f'{database}.{collection}' not in collections:
        collections[f'{database}.{collection}'] = client[database][collection]
    collection = collections[f'{database}.{collection}']
    collection.remote = False
    collection.unset_hash_set()
    return make_response('ok', 200)


@app.route('/clear_remote_cache', methods=['PUT'])
def clear_cache():
    keys = list(collections.keys())[:]
    for k in keys:
        del collections[k]
    return make_response('ok', 200)


@app.route('/count/<database>/<collection>', methods=['GET'])
def count(database, collection):
    collection = collections[f'{database}.{collection}']
    print(collection.database.models)
    return make_response(str(collection.count_documents({})), 200)


@app.route('/find_nearest', methods=['GET'])
def find_nearest():
    data = request.get_json()
    database = data['database']
    collection = data['collection']
    semantic_index = data['semantic_index']
    if 'filter' in data:
        filter = BSON.decode(data['filter'].encode('iso-8859-1'))
    else:
        filter = {}
    if 'ids' not in data:
        ids = None
    else:
        ids = [ObjectId(_id) for _id in data['ids']]
    if f'{database}.{collection}' not in collections:
        collections[f'{database}.{collection}'] = client[database][collection]
    collection = collections[f'{database}.{collection}']
    collection.remote = False
    from superduperdb.types.utils import convert_from_bytes_to_types
    filter = convert_from_bytes_to_types(filter, converters=collection.types)
    result = collection._find_nearest(filter, ids=ids, semantic_index=semantic_index)
    for i, _id in enumerate(result['_ids']):
        result['_ids'][i] = str(_id)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='localhost', port=cf['linear_algebra']['port'])
