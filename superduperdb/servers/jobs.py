import datetime

from bson import ObjectId
import uuid

from flask import Flask, request
from redis import Redis
from rq import Queue

from superduperdb.client import the_client
from superduperdb.jobs import process as process_jobs
from superduperdb import cf

q = Queue(connection=Redis(port=cf['redis']['port']), default_timeout=24 * 60 * 60)
app = Flask(__name__)


@app.route('/process/', methods=['POST'])
def process():
    data = dict(request.get_json())
    assert 'collection' in data
    assert 'database' in data
    assert 'method' in data
    assert 'kwargs' in data
    if 'ids' in data['kwargs']:
        data['kwargs']['ids'] = [ObjectId(id_) for id_ in data['kwargs']['ids']]
    job_id = str(uuid.uuid4())
    collection = the_client[data['database']][data['collection']]
    collection['_jobs'].insert_one({
        'identifier': job_id,
        'time': datetime.datetime.now(),
        'status': 'pending',
        'method': data['method'],
        'args': data.get('args', ()),
        'kwargs': data['kwargs'],
        'stdout': [],
        'stderr': [],
    })
    job = q.enqueue(
        process_jobs._function_job,
        data['database'],
        data['collection'],
        data['method'],
        job_id,
        job_id=job_id,
        args_=data.get('args', ()),
        kwargs_=data['kwargs'],
        depends_on=data.get('dependencies', ())
    )
    return str(job.id)


if __name__ == '__main__':
    app.run(host='localhost', port=cf['jobs']['port'])

