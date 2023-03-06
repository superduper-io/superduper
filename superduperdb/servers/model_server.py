import pickle

import flask

from superduperdb.client import SuperDuperClient
from flask import request, Flask

# https://flask.palletsprojects.com/en/2.1.x/patterns/streaming/ streaming for the find endpoint

from superduperdb import cf

app = Flask(__name__)

client = SuperDuperClient(**cf['mongodb'])
collections = {}


@app.route('/apply_model', methods=['GET'])
def apply_model():
    data = request.get_json()
    database = data['database']
    collection = data['collection']
    name = data['name']
    kwargs = data.get('kwargs', {})
    input_ = pickle.loads(data['input_'].encode('iso-8859-1'))
    if f'{database}.{collection}' not in collections:
        collections[f'{database}.{collection}'] = client[database][collection]
    print(input_)
    collection = collections[f'{database}.{collection}']
    collection.remote = False
    result = collection.apply_model(name, input_, **kwargs)
    result = collection.convert_types(result)
    response = flask.make_response(pickle.dumps(result))
    return response


if __name__ == '__main__':
    app.run(host='localhost', port=cf['model_server']['port'])