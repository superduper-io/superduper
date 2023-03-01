import datetime

import uuid

from redis import Redis
from rq import Queue

from superduperdb.client import the_client
from superduperdb.jobs import process as process_jobs
from superduperdb import cf

q = Queue(connection=Redis(port=cf['redis']['port']), default_timeout=24 * 60 * 60)


def process(database, collection, method, *args, dependencies=(), **kwargs):
    job_id = str(uuid.uuid4())
    collection = the_client[database][collection]
    collection['_jobs'].insert_one({
        'identifier': job_id,
        'time': datetime.datetime.now(),
        'status': 'pending',
        'method': method,
        'args': args,
        'kwargs': kwargs,
        'stdout': [],
        'stderr': [],
    })
    job = q.enqueue(
        process_jobs._function_job,
        database,
        collection,
        method,
        job_id,
        job_id=job_id,
        args_=args,
        kwargs_=kwargs,
        depends_on=dependencies,
    )
    return str(job.id)
