import datetime
import inspect
from functools import wraps
import traceback
import uuid

from rq import Queue
from redis import Redis

from superduperdb import cf
from superduperdb.cluster.annotations import encode_args, encode_kwargs, decode_args, decode_kwargs
from superduperdb.cluster.process import handle_function_output
from superduperdb.base.imports import get_database_from_database_type

redis_cf = cf.get('redis', {'port': 6379, 'host': 'localhost'})

q = Queue(connection=Redis(port=cf.get('redis', {}).get('port', 6379)), default_timeout=24 * 60 * 60)


def work(f):
    sig = inspect.signature(f)
    @wraps(f)
    def work_wrapper(database, *args, remote=None, dependencies=(), **kwargs):
        if remote is None:
            remote = database.remote
        if remote:
            args = encode_args(database, sig, args)
            kwargs = encode_kwargs(database, sig, kwargs)
            job_id = str(uuid.uuid4())
            database._create_job_record({
                'identifier': job_id,
                'time': datetime.datetime.now(),
                'status': 'pending',
                'method': f.__name__,
                'args': args,
                'kwargs': kwargs,
                'stdout': [],
                'stderr': [],
            })
            kwargs['remote'] = False
            job = q.enqueue(
                function_job,
                database._database_type,
                database.name,
                f.__name__,
                args,
                kwargs,
                job_id,
                job_id=job_id,
                depends_on=dependencies,
            )
            return job
        else:
            return f(database, *args, **kwargs)
    work_wrapper.signature = sig
    return work_wrapper


def function_job(database_type, database_name, function_name,
                 args_, kwargs_, task_id):
    database = get_database_from_database_type(database_type, database_name)
    database.remote = False
    function = getattr(database, function_name)
    database.set_job_flag(task_id, ('status', 'running'))
    args_ = decode_args(database, function.signature, args_)
    kwargs_ = decode_kwargs(database, function.signature, kwargs_)
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

