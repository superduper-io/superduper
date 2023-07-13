from bson import ObjectId
import uuid

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
    dependencies = data.get('dependencies', ())
    print(dependencies)
    job_id = str(uuid.uuid4())
    job = q.enqueue(
        process.process_documents_with_model,
        database=database,
        collection=collection,
        ids=ids,
        model_name=model_name,
        batch_size=batch_size,
        verbose=verbose,
        identifier=job_id,
        job_id=job_id,
        depends_on=dependencies,
    )
    if blocking:
        while True:
            if job.get_status(refresh=True) in {'finished', 'stopped', 'canceled', 'failed'}:
                break
    return str(job.id)


@app.route('/download_content', methods=['POST'])
def download_content():
    data = request.get_json()
    blocking = data.get('blocking', False)
    database = data.get('database')
    collection = data.get('collection')
    ids = data.get('ids')
    ids = [ObjectId(id_) for id_ in ids]
    dependencies = data.get('dependencies', ())
    job_id = str(uuid.uuid4())
    job = q.enqueue(process.download_content, database=database, collection=collection, ids=ids,
                    identifier=job_id, job_id=job_id, depends_on=dependencies)
    if blocking:
        while True:
            if job.get_status(refresh=True) in {'finished', 'stopped', 'canceled', 'failed'}:
                break
    return str(job.id)


if __name__ == '__main__':
    app.run(host='localhost', port=cf['jobs']['port'])

