from bson import ObjectId
import json

from flask import Flask, request
from redis import Redis
from rq import Queue

from sddb.jobs.process import process_documents_with_model


q = Queue(connection=Redis())
app = Flask(__name__)


@app.route('/process_documents_with_model', methods=['GET'])
def process_documents_with_model():
    database = request.args.get('database')
    collection = request.args.get('collection')
    ids = json.loads(request.args.get('ids'))
    ids = [ObjectId(id_) for id_ in ids]
    model_name = request.args.get('model_name')
    batch_size = int(request.args.get('batch_size'))
    verbose = request.args.get('verbose').lower() == 'true'
    q.enqueue(process_documents_with_model,
              database=database,
              collection=collection,
              ids=ids,
              model_name=model_name,
              batch_size=batch_size,
              verbose=verbose)


if __name__ == '__main__':
    app.run(host='localhost', port=5001)

