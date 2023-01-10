from superduperdb.client import SuperDuperClient
from bson import BSON, ObjectId
from flask import request, Flask, jsonify

# https://flask.palletsprojects.com/en/2.1.x/patterns/streaming/ streaming for the find endpoint

from superduperdb import cf

app = Flask(__name__)

client = SuperDuperClient(**cf['mongodb'])
collections = {}


@app.route('/_find_nearest', methods=['GET'])
def _find_nearest():
    data = request.get_json()
    database = data['database']
    collection = data['collection']
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
    collection.single_thread = True
    from superduperdb.types.utils import convert_types
    filter = convert_types(filter, converters=collection.converters)
    result = collection._find_nearest(filter, ids=ids)
    for i, _id in enumerate(result['_ids']):
        result['_ids'][i] = str(_id)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='localhost', port=cf['master']['port'])
