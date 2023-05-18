from bson import BSON
from functools import wraps
import inspect
import requests

from superduperdb import cf
from superduperdb.cluster.annotations import (
    encode_args,
    encode_kwargs,
    decode_result,
)
from superduperdb.misc.logger import logging


def use_vector_search(
    database_type, database_name, table_name, method, args, kwargs
):
    bson_ = {
        'database_name': database_name,
        'database_type': database_type,
        'table': table_name,
        'method': method,
        'args': args,
        'kwargs': kwargs,
    }
    body = BSON.encode(bson_)
    response = requests.post(
        f'http://{cf["model_server"]["host"]}:{cf["vector_search"]["port"]}/',
        data=body,
    )
    out = BSON.decode(response.content)
    return out['_out']


def vector_search(f):
    sig = inspect.signature(f)

    @wraps(f)
    def vector_search_wrapper(table, *args, remote=None, **kwargs):
        if remote is None:
            remote = table.remote
        if remote:
            args = encode_args(table.database, sig, args)
            kwargs = encode_kwargs(table.database, sig, kwargs)
            out = use_vector_search(
                table.database._database_type,
                table.database.name,
                table.name,
                f.__name__,
                args,
                kwargs,
            )
            out = decode_result(table.database, sig, out)
            return out
        else:
            return f(table, *args, **kwargs)

    vector_search_wrapper.signature = sig
    vector_search_wrapper.f = f
    return vector_search_wrapper


def model_server(f):
    """
    Method decorator to posit that function is called on the remote, not on the client.

    :param f: method object
    """

    sig = inspect.signature(f)

    @wraps(f)
    def model_server_wrapper(database, *args, remote=None, **kwargs):
        if remote is None:
            remote = database.remote
        if remote:
            args = encode_args(database, sig, args)
            kwargs = encode_kwargs(database, sig, kwargs)
            out = use_model_server(
                database._database_type,
                database.name,
                f.__name__,
                args,
                kwargs,
            )
            out = decode_result(database, sig, out)
            return out
        else:
            return f(database, *args, **kwargs)

    model_server_wrapper.f = f
    return model_server_wrapper


RAY_MODELS = [
    (r['database'], r['model'])
    for r in cf.get('ray', {}).get('deployments', [])
]

logging.info(f'These are the RAY_MODELS: {RAY_MODELS}')


def use_model_server(database_type, database_name, method, args, kwargs):
    if (
        method in {'predict', 'predict_one'}
        and (database_name, args[0]) in RAY_MODELS
    ):
        logging.debug('using Ray server')
        response = requests.post(
            f'http://{cf["ray"]["host"]}:{cf["ray"]["port"]}/{method}/{args[0]}',
            data=BSON.encode({'input_': args[1]}),
        )
    else:
        bson_ = {
            'database_type': database_type,
            'database_name': database_name,
            'method': method,
            'args': args,
            'kwargs': kwargs,
        }
        body = BSON.encode(bson_)
        response = requests.post(
            f'http://{cf["model_server"]["host"]}:{cf["model_server"]["port"]}/',
            data=body,
        )
    out = BSON.decode(response.content)['output']
    return out
