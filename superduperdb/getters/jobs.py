import datetime

import uuid

from redis import Redis
from rq import Queue

from superduperdb.jobs import process as process_jobs
from superduperdb import cf

q = Queue(connection=Redis(port=cf['redis']['port']), default_timeout=24 * 60 * 60)


def process(database_name, collection_name, method, *args, dependencies=(), **kwargs):
    from superduperdb.client import the_client
    job_id = str(uuid.uuid4())
    collection = the_client[database_name][collection_name]
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
        database_name,
        collection_name,
        method,
        job_id,
        job_id=job_id,
        args_=args,
        kwargs_=kwargs,
        depends_on=dependencies,
    )
    return str(job.id)
