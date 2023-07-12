from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash

from superduperdb.datalayer.base.build import build_datalayer
from bson import BSON
from flask import request, Flask, make_response

from superduperdb import CFG
from superduperdb.cluster.login import maybe_login_required
from superduperdb.misc.logger import logging

app = Flask(__name__)
CORS(app)
auth = HTTPBasicAuth()

if CFG.vector_search.username:
    password_hash = generate_password_hash(CFG.vector_search.password)
    users = {CFG.vector_search.username: password_hash}
else:
    users = None

database = build_datalayer()


@app.route('/', methods=['POST'])
@maybe_login_required(auth, 'vector_search')
def serve():
    data = BSON.decode(request.get_data())
    database.distributed = False
    method = getattr(database, data['method'])
    result = method.f(database, *data['args'], **data['kwargs'])
    logging.info('results')
    logging.info(result)
    logging.info(result)
    return make_response(BSON.encode({'_out': result}))


if __name__ == '__main__':
    app.run(host='localhost', port=CFG.vector_search.port)
