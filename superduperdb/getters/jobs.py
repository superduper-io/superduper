import datetime

import uuid

import subprocess
from redis import Redis
from rq import Queue, Worker
from rq.job import Job

from superduperdb.jobs import process as process_jobs
from superduperdb import cf
from superduperdb.utils import get_database_from_database_type

# by default insecure on localhost - add user/ password for security
redis_cf = cf.get('redis', {'port': 6379, 'host': 'localhost'})

redis = Redis(**redis_cf)
q = Queue(connection=redis, default_timeout=24 * 60 * 60)


def stop_job(database_type, database_name, job_id):
    database = get_database_from_database_type(database_type, database_name)
    job = Job.fetch(job_id, connection=redis)
    job.cancel()
    worker = Worker.find_by_key(job.worker_name)
    subprocess.call(['kill', str(worker.pid)])
    database._update_job_info({'identifier': job_id}, 'status', 'aborted')


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
        database_type,
        database_name,
        method,
        job_id,
        job_id=job_id,
        args_=args,
        kwargs_=kwargs,
        depends_on=dependencies,
    )
    return str(job.id)
