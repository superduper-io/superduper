import datetime
import traceback

import uuid

import subprocess
from redis import Redis
from rq import Queue, Worker
from rq.job import Job

from celery import Celery, chain
from celery.result import AsyncResult

from superduperdb.jobs import process as process_jobs
from superduperdb import cf
from superduperdb.jobs.process import handle_function_output
from superduperdb.utils import get_database_from_database_type

# by default insecure on localhost - add user/ password for security
redis_cf = cf.get('redis', {'port': 6379, 'host': 'localhost'})

# redis = Redis(**redis_cf)
# q = Queue(connection=redis, default_timeout=24 * 60 * 60)

app = Celery('superduperdb.getters.jobs', broker=f'redis://{redis_cf["host"]}:{redis_cf["port"]}/0')

# def stop_job(database_type, database_name, job_id):
#     database = get_database_from_database_type(database_type, database_name)
#     job = Job.fetch(job_id, connection=redis)
#     job.cancel()
#     worker = Worker.find_by_key(job.worker_name)
#     subprocess.call(['kill', str(worker.pid)])
#     database._update_job_info({'identifier': job_id}, 'status', 'aborted')


def create_task_graph_from_graph():
    ...


def stop_job(database_type, database_name, job_id):
    database = get_database_from_database_type(database_type, database_name)
    app.control.revoke(job_id, terminate=True)
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
    if not dependencies:
        result = function_job.apply_async(
            args=[
                database_type,
                database_name,
                method,
                args,
                kwargs,
                job_id,
            ],
            task_id=job_id
        )
    else:
        result = chain(
            *dependencies,
            function_job.signature(
                args=[database_type, database_name, method, args, kwargs, job_id]
            ),
            task_id=job_id
        ).apply_async()
    return result


@app.task
def function_job(database_type, database_name, function_name,
                 args_, kwargs_, task_id):
    database = get_database_from_database_type(database_type, database_name)
    database.remote = False
    function = getattr(database, function_name)
    database.set_job_flag(task_id, ('status', 'running'))
    try:
        handle_function_output(
            function,
            database,
            task_id,
            *args_,
            **kwargs_,
        )
    except Exception as e:
        tb = traceback.format_exc()
        database.set_job_flag(task_id, ('status', 'failed'))
        database.set_job_flag(task_id, ('msg', tb))
        raise e
    database.set_job_flag(task_id, ('status', 'success'))


if __name__ == '__main__':
    pass