import flask
from bson import BSON
from flask_cors import CORS
from flask import request, Flask
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from superduperdb.mongodb.client import SuperDuperClient
from superduperdb import cf
from superduperdb.serving.utils import maybe_login_required, decode_args_kwargs
from superduperdb.utils import ArgumentDefaultDict

app = Flask(__name__)
CORS(app)
auth = HTTPBasicAuth()

if 'user' in cf['model_server']:
    users = {
        cf['model_server']['user']: generate_password_hash(cf['model_server']['password']),
    }

client = SuperDuperClient(**cf['mongodb'])
databases = ArgumentDefaultDict(lambda name: client[name])


@auth.verify_password
def verify_password(username, password):
    if username in users and \
            check_password_hash(users.get(username), password):
        return username


@app.route('/', methods=['POST'])
@maybe_login_required(auth, 'model_server')
def serve():
    data = BSON.decode(request.get_data())
    database = data['database']
    if f'{database}' not in databases:
        databases[f'{database}'] = client[database]
    database = databases[data['database']]
    method = getattr(database, data['method'])
    args, kwargs = decode_args_kwargs(database,
                                      data['args'],
                                      {k: v for k, v in data['kwargs'].items() if k != 'remote'},
                                      method.positional_convertible,
                                      method.keyword_convertible)
    result = method(*args, **kwargs, remote=False)
    if method.return_convertible:
        result = database.convert_from_types_to_bytes(result)
    return flask.make_response(BSON.encode(result))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=cf['model_server']['port'])