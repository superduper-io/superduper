from flask import Flask, jsonify, request
from sddb.client import SddbClient
from sddb.models.converters import FloatTensor
from sddb import cf

app = Flask(__name__)

client = SddbClient(**cf['mongodb'])
collections = {}


@app.route('/find_nearest_from_hash', methods=['POST'])
def find_nearest_from_hash():
    data = request.get_json()
    database = data.get('database')
    collection = data.get('collection')
    semantic_index = data.get('semantic_index')
    n = int(data.get('n'))
    h = FloatTensor.decode(data.get('h').encode('iso-8859-1'))
    if f'{database}.{collection}' not in collections:
        collections[f'{database}.{collection}'] = client[database][collection]
    collection = collections[f'{database}.{collection}']
    collection.single_thread = True
    output = collection.find_nearest_from_hash(semantic_index, h, n=n)
    output['ids'] = [str(id_) for id_ in output['ids']]
    return jsonify(output)


if __name__ == '__main__':
    app.run(host='localhost', port=5000)

