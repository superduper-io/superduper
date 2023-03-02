from superduperdb.client import SuperDuperClient
from bson import BSON, ObjectId
from flask import request, Flask, jsonify

# https://flask.palletsprojects.com/en/2.1.x/patterns/streaming/ streaming for the find endpoint

from superduperdb import cf

app = Flask(__name__)

client = SuperDuperClient(**cf['mongodb'])
collections = {}


@app.route('/_apply_model', methods=['GET'])
def _apply_model():
    data = request.get_json()
    database = data['database']
    collection = data['collection']
    name = data['name']
    input_ = data['input_']
    kwargs = data.get('kwargs', {})
    if f'{database}.{collection}' not in collections:
        collections[f'{database}.{collection}'] = client[database][collection]
    collection = collections[f'{database}.{collection}']
    collection.remote = False
    from superduperdb.types.utils import convert_types
    input_ = convert_types(input_, collection.types)
    result = collection.apply_model(name, input_, **kwargs)
    return BSON.encode(result)


if __name__ == '__main__':
    app.run(host='localhost', port=cf['linear_algebra']['port'])