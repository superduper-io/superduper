from sddb.client import SddbClient
from bson import BSON
from flask import request, Flask

# https://flask.palletsprojects.com/en/2.1.x/patterns/streaming/ streaming for the find endpoint

from sddb import cf

app = Flask(__name__)

client = SddbClient(**cf['mongodb'])
collections = {}


@app.route('/find', methods=['GET'])
def find():
    data = request.get_json()
    database = data['database']
    collection = data['collection']
    args = data['args']
    kwargs = data['kwargs']
    if f'{database}.{collection}' not in collections:
        collections[f'{database}.{collection}'] = client[database][collection]
    collection = collections[f'{database}.{collection}']
    collection.remote = False
    result = collection.find(*args, convert=False, **kwargs)
    def generate():
        for r in result:
            yield bytes(BSON.encode(r))
    return app.response_class(generate(), mimetype='text/csv')


@app.route('/find_one', methods=['GET'])
def find_one():
    data = request.get_json()
    database = data['database']
    collection = data['collection']
    args = data['args']
    if 'filter' in data:
        filter = BSON.decode(data['filter'].encode('iso-8859-1'))
    else:
        filter = {}
    kwargs = data['kwargs']
    if f'{database}.{collection}' not in collections:
        collections[f'{database}.{collection}'] = client[database][collection]
    collection = collections[f'{database}.{collection}']
    collection.remote = False
    collection.single_thread = True
    result = collection.find_one(filter, *args, convert=False, **kwargs)
    return bytes(BSON.encode(result))


if __name__ == '__main__':
    app.run(host='localhost', port=cf['master']['port'])
