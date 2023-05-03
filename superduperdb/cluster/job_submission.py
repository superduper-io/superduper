import datetime
import inspect
from functools import wraps
import uuid

from rq import Queue
from redis import Redis

from superduperdb import cf
from superduperdb.cluster.annotations import encode_args, encode_kwargs
from superduperdb.cluster.function_job import function_job

ENGINE = cf.get('job_engine', 'rq')


if ENGINE == 'rq':
    redis_cf = cf.get('redis', {'port': 6379, 'host': 'localhost'})
    q = Queue(connection=Redis(port=cf.get('redis', {}).get('port', 6379)), default_timeout=24 * 60 * 60)
elif ENGINE == 'dask':
    from superduperdb.cluster.dask.dask_client import dask_client
else:
    raise NotImplementedError(f'The engine {ENGINE} has not been implemented...')


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
            if ENGINE == 'rq':
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
            elif ENGINE == 'dask':
                job = dask_client.submit(
                    function_job,
                    database._database_type,
                    database.name,
                    f.__name__,
                    args,
                    kwargs,
                    job_id,
                )
            else:
                raise NotImplementedError(f'That engine: {ENGINE} hasn\'t been implemented...')
            return job
        else:
            print(database)
            print(args)
            print(kwargs)
            return f(database, *args, **kwargs)
    work_wrapper.signature = sig
    return work_wrapper

