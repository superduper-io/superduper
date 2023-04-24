import inspect

from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash

from superduperdb.base.imports import get_database_from_database_type
from superduperdb.cluster.annotations import decode_args, decode_kwargs, encode_result
from superduperdb.mongodb.client import SuperDuperClient
from bson import BSON
from flask import request, Flask, make_response

from superduperdb import cf
from superduperdb.cluster.login import maybe_login_required

app = Flask(__name__)
CORS(app)
auth = HTTPBasicAuth()

if 'user' in cf['vector_search']:
    users = {
        cf['vector_search']['user']: generate_password_hash(cf['vector_search']['password']),
    }

client = SuperDuperClient(**cf['mongodb'])
collections = {}


@app.route('/', methods=['POST'])
@maybe_login_required(auth, 'vector_search')
def serve():
    data = BSON.decode(request.get_data())
    database = get_database_from_database_type(data['database_type'], data['database_name'])
    table = getattr(database, data['table'])
    method = getattr(table, data['method'])
    table.remote = False
    args = decode_args(database,
                       inspect.signature(method),
                       data['args'])
    kwargs = decode_kwargs(database,
                           inspect.signature(method),
                           data['kwargs'])
    result = method(*args, **kwargs)
    result = encode_result(database, result)
    if isinstance(result, tuple):
        return make_response(BSON.dumps({'_out': result}))


if __name__ == '__main__':
    app.run(host='localhost', port=cf['linear_algebra']['port'])
