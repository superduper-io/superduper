import datetime
import inspect
from functools import wraps
import traceback
import uuid
from typing import List

from bson import ObjectId
from rq import Queue
from redis import Redis

from superduperdb import cf
from superduperdb.cluster.process import handle_function_output
from superduperdb.cluster.utils import encode_ids_parameters
from superduperdb.database import get_database_from_database_type

redis_cf = cf.get('redis', {'port': 6379, 'host': 'localhost'})

q = Queue(connection=Redis(port=cf.get('redis', {}).get('port', 6379)), default_timeout=24 * 60 * 60)


def get_ids_parameters(f):
    """
    Get parameters which need to be converted from/ to ObjectIds

    :param f: method object
    """
    sig = inspect.signature(f)
    parameters = sig.parameters
    positional_parameters = [k for k in parameters
                             if parameters[k].default == inspect.Parameter.empty
                             and parameters[k].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                             and k != 'self']
    keyword_parameters = [k for k in parameters if k not in set(positional_parameters)
                          and parameters[k].default != inspect.Parameter.empty
                          and parameters[k].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
    positional_convertible = \
        [parameters[k].annotation == List[ObjectId] for k in positional_parameters]
    keyword_convertible = {k: parameters[k].annotation == List[ObjectId]
                           for k in keyword_parameters}
    return positional_convertible, keyword_convertible


def work(f):
    f.positional_convertible, f.keyword_convertible = get_ids_parameters(f)
    @wraps(f)
    def work_wrapper(database, *args, remote=None, dependencies=(), **kwargs):
        if remote is None:
            remote = database.remote
        if remote:
            args, kwargs = encode_ids_parameters(args, kwargs, f.positional_convertible,
                                                 f.keyword_convertible)
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
    return work_wrapper


def decode_ids_parameters(args, kwargs, positional_convertible, keyword_convertible):
    args = list(args)
    for i, arg in enumerate(args):
        if positional_convertible[i]:
            args[i] = [ObjectId(_id) for _id in arg]

    for k, v in kwargs.items():
        if keyword_convertible[k]:
            kwargs[k] = [ObjectId(_id) for _id in v]

    return args, kwargs


def function_job(database_type, database_name, function_name,
                 args_, kwargs_, task_id):
    database = get_database_from_database_type(database_type, database_name)
    database.remote = False
    function = getattr(database, function_name)
    database.set_job_flag(task_id, ('status', 'running'))
    args_, kwargs_ = decode_ids_parameters(
        args_,
        {k: v for k, v in kwargs_.items() if k not in {'remote', 'do_work'}},
        function.positional_convertible, function.keyword_convertible

    )
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

