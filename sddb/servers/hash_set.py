import base64
from flask import Flask, jsonify, request
from sddb.client import SddbClient
from sddb.models.converters import FloatTensor

app = Flask(__name__)

client = SddbClient()
collections = {}


@app.route('/find_nearest_from_hash', methods=['GET'])
def find_nearest_from_hash():
    database = request.args.get('database')
    collection = request.args.get('collection')
    semantic_index = request.args.get('semantic_index')
    n = int(request.args.get('n'))
    h = FloatTensor.decode(base64.b64decode(request.args.get('h')))
    if f'{database}.{collection}' not in collections:
        collections[f'{database}.{collection}'] = client[database][collection]
    collection = collections[f'{database}.{collection}']
    output = collection.find_nearest_from_hash(semantic_index, h, n=n)
    output['ids'] = [str(id_) for id_ in output['ids']]
    return jsonify(output)


if __name__ == '__main__':
    app.run(host='localhost', port=5000)

