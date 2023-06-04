import inspect

from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash

from superduperdb.datalayer.base.imports import get_database_from_database_type
from superduperdb.cluster.annotations import decode_args, decode_kwargs, encode_result
from superduperdb.datalayer.mongodb.client import SuperDuperClient
from bson import BSON
from flask import request, Flask, make_response

from superduperdb import CFG
from superduperdb.cluster.login import maybe_login_required
from superduperdb.misc.logger import logging

app = Flask(__name__)
CORS(app)
auth = HTTPBasicAuth()

if CFG.vector_search.user:
    password_hash = generate_password_hash(CFG.vector_search.password)
    users = {CFG.vector_search.user: password_hash}
else:
    users = None

client = SuperDuperClient(**CFG.mongodb.dict())
collections = {}


@app.route('/', methods=['POST'])
@maybe_login_required(auth, 'vector_search')
def serve():
    data = BSON.decode(request.get_data())
    database = get_database_from_database_type(
        data['database_type'], data['database_name']
    )
    database.remote = False
    method = getattr(database, data['method'])
    args = decode_args(database, inspect.signature(method), data['args'])
    kwargs = decode_kwargs(database, inspect.signature(method), data['kwargs'])
    result = method.f(database, *args, **kwargs)
    logging.info('results')
    logging.info(result)
    result = encode_result(database, method.signature, result)
    logging.info(result)
    return make_response(BSON.encode({'_out': result}))


if __name__ == '__main__':
    app.run(host='localhost', port=CFG.vector_search.port)
