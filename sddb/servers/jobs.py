from bson import ObjectId

from flask import Flask, request
from redis import Redis
from rq import Queue

from sddb.jobs import process
from sddb import cf

q = Queue(connection=Redis(port=cf['redis']['port']))
app = Flask(__name__)


@app.route('/process_documents_with_model', methods=['POST'])
def process_documents_with_model():
    data = request.get_json()
    database = data.get('database')
    collection = data.get('collection')
    ids = data.get('ids')
    ids = [ObjectId(id_) for id_ in ids]
    model_name = data.get('model_name')
    batch_size = int(data.get('batch_size'))
    verbose = data.get('verbose')
    blocking = data.get('blocking', False)
    job = q.enqueue(
        process.process_documents_with_model,
        database=database,
        collection=collection,
        ids=ids,
        model_name=model_name,
        batch_size=batch_size,
        verbose=verbose,
    )
    if blocking:
        while True:
            if job.get_status(refresh=True) in {'finished', 'stopped', 'canceled', 'failed'}:
                break


if __name__ == '__main__':
    app.run(host='localhost', port=5001)

