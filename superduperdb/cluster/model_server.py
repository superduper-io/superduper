import flask
from bson import BSON
from flask_cors import CORS
from flask import request, Flask
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from superduperdb.datalayer.mongodb.client import SuperDuperClient
from superduperdb import CFG
from superduperdb.cluster.login import maybe_login_required
from superduperdb.misc.special_dicts import ArgumentDefaultDict

app = Flask(__name__)
CORS(app)
auth = HTTPBasicAuth()

if CFG.model_server.user:
    password_hash = generate_password_hash(CFG.model_server.password)
    users = {CFG.model_server.user: password_hash}
else:
    users = None

client = SuperDuperClient(**CFG.mongodb.dict())
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
    result = method(database, *data['args'], **data['kwargs'])
    return flask.make_response(BSON.encode(result))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=CFG.model_server.port)
