import datetime

import uuid

from redis import Redis
from rq import Queue

from superduperdb.jobs import process as process_jobs
from superduperdb import cf
from superduperdb.utils import get_database_from_database_type

q = Queue(connection=Redis(port=cf.get('redis', {}).get('port', 6379)), default_timeout=24 * 60 * 60)


def process(database_type, database_name, method, *args, dependencies=(),
            **kwargs):
    job_id = str(uuid.uuid4())
    database = get_database_from_database_type(database_type, database_name)
    database._create_job_record({
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
        method,
        job_id,
        job_id=job_id,
        args_=args,
        kwargs_=kwargs,
        depends_on=dependencies,
    )
    return str(job.id)
