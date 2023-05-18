import inspect

import flask
from bson import BSON
from flask_cors import CORS
from flask import request, Flask
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from superduperdb.cluster.annotations import (
    decode_args,
    decode_kwargs,
    encode_result,
)
from superduperdb.datalayer.mongodb.client import SuperDuperClient
from superduperdb import cf
from superduperdb.cluster.login import maybe_login_required
from superduperdb.misc.special_dicts import ArgumentDefaultDict

app = Flask(__name__)
CORS(app)
auth = HTTPBasicAuth()

if 'user' in cf['model_server']:
    users = {
        cf['model_server']['user']: generate_password_hash(
            cf['model_server']['password']
        ),
    }

client = SuperDuperClient(**cf['mongodb'])
databases = ArgumentDefaultDict(lambda name: client[name])


@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username


@app.route('/', methods=['POST'])
@maybe_login_required(auth, 'model_server')
def serve():
    data = BSON.decode(request.get_data())
    database = data['database_name']
    if f'{database}' not in databases:
        databases[f'{database}'] = client[database]
    database = databases[database]
    method = getattr(database, data['method']).f
    args = decode_args(database, inspect.signature(method), data['args'])
    kwargs = decode_kwargs(database, inspect.signature(method), data['kwargs'])
    result = method(database, *args, **kwargs)
    result = encode_result(
        database, inspect.signature(method), {'output': result}
    )
    return flask.make_response(BSON.encode(result))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=cf['model_server']['port'])
