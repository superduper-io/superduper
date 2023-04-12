import pickle
import flask
from flask_cors import CORS
from flask import request, Flask
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from superduperdb.mongodb.client import SuperDuperClient
from superduperdb import cf
from superduperdb.servers.utils import maybe_login_required

app = Flask(__name__)
CORS(app)
auth = HTTPBasicAuth()

if 'user' in cf['model_server']:
    users = {
        cf['model_server']['user']: generate_password_hash(cf['model_server']['password']),
    }

client = SuperDuperClient(**cf['mongodb'])
databases = {}


@auth.verify_password
def verify_password(username, password):
    if username in users and \
            check_password_hash(users.get(username), password):
        return username


@app.route('/apply_model', methods=['GET'])
@maybe_login_required(auth, 'model_server')
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
    result = database.convert_from_types_to_bytes(result)
    response = flask.make_response(pickle.dumps(result))
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=cf['model_server']['port'])