import pickle
import flask

from superduperdb.mongodb.client import SuperDuperClient
from flask import request, Flask

# TODO - use torchserve...

# https://flask.palletsprojects.com/en/2.1.x/patterns/streaming/ streaming for the find endpoint

from superduperdb import cf

app = Flask(__name__)

client = SuperDuperClient(**cf['mongodb'])
databases = {}


@app.route('/apply_model', methods=['GET'])
def apply_model():
    data = request.get_json()
    database = data['database']
    name = data['name']
    kwargs = data.get('kwargs', {})
    input_ = pickle.loads(data['input_'].encode('iso-8859-1'))
    if f'{database}' not in databases:
        databases[f'{database}'] = client[database]
    print(input_)
    database = databases[f'{database}']
    database.remote = False
    result = database.apply_model(name, input_, **kwargs)
    result = database.convert_types(result)
    response = flask.make_response(pickle.dumps(result))
    return response


if __name__ == '__main__':
    app.run(host='localhost', port=cf['model_server']['port'])