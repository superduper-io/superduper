import inspect
from functools import wraps

from bson import BSON
import requests

from superduperdb import cf
from superduperdb.cluster.utils import encode_args_kwargs
from superduperdb.types.utils import convert_from_bytes_to_types
from superduperdb.database import get_database_from_database_type


def get_convertible_parameters(f):
    """
    Get parameters which need to be converted from a instance method.

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
        [parameters[k].annotation == Convertible for k in positional_parameters]
    keyword_convertible = {k: parameters[k].annotation == Convertible
                           for k in keyword_parameters}
    return_convertible = sig.return_annotation == Convertible
    return positional_convertible, keyword_convertible, return_convertible


class Convertible:
    """
    Type annotation to denote that a parameter has content which needs converting.
    """
    ...


databases = {}


def use_vector_search(database_type, database_name, method, args, kwargs):
    if database_name not in databases:
        database = get_database_from_database_type(database_type, database_name)
        databases[database_name] = database
    bson_ = {
        'database': database_name,
        'method': method,
        'args': args,
        'kwargs': {
            **kwargs,
            'remote': False,
        },
    }
    body = BSON.encode(bson_)
    response = requests.post(
        f'http://{cf["model_server"]["host"]}:{cf["model_server"]["port"]}/',
        data=body,
    )
    out = BSON.decode(response.content)
    return out


def vector_search(f):
    f.positional_convertible, f.keyword_convertible, f.return_convertible = \
        get_convertible_parameters(f)

    @wraps(f)
    def vector_search_wrapper(database, *args, remote=cf['remote'], **kwargs):
        args, kwargs = encode_args_kwargs(database, args, kwargs, f.positional_convertible,
                                          f.keyword_convertible)
        if remote:
            out = use_vector_search(
                f.__self__._database_type,
                f.__self__.name,
                f.__name__,
                args,
                kwargs,
            )
            return out
        else:
            return f(*args, **kwargs, remote=False)

    return vector_search_wrapper


def model_server(f):
    """
    Method decorator to posit that function is called on the remote, not on the client.

    :param f: method object
    """

    f.positional_convertible, f.keyword_convertible, f.return_convertible = \
        get_convertible_parameters(f)
    @wraps(f)
    def model_server_wrapper(database, *args, remote=None, **kwargs):
        if remote is None:
            remote = database.remote
        if remote:
            args, kwargs = encode_args_kwargs(database, args, kwargs, f.positional_convertible,
                                              f.keyword_convertible)
            out = use_model_server(
                database._database_type,
                database.name,
                f.__name__,
                args,
                kwargs,
            )
            if f.return_convertible:
                out = convert_from_bytes_to_types(out, database.types)
            return out
        else:
            return f(database, *args, **kwargs, remote=False)
    return model_server_wrapper


def find_nearest(database, collection, like, ids=None, semantic_index=None):
    """
    Send off request to remote to find-nearest item based on ``like`` item and ``semantic_index``
    Supports MongoDB.

    :param database: name of database
    :param collection: name of collection
    :param like: name of like
    :param ids: name of ids
    :param semantic_index: name of semantic-index
    """
    bson_ = {
        'database': database,
        'collection': collection,
        'filter': like,
        'semantic_index': semantic_index,
        'ids': ids,
    }
    body = BSON.encode(bson_)
    response = requests.post(
        f'http://{cf["linear_algebra"]["host"]}:{cf["linear_algebra"]["port"]}/find_nearest',
        headers=None,
        stream=True,
        data=body,
    )
    results = BSON.decode(response.content)
    return results


def clear_remote_cache():
    response = requests.put(
        f'http://{cf["linear_algebra"]["host"]}:{cf["linear_algebra"]["port"]}/clear_remote_cache',
        headers=None,
    )
    if response.status_code != 200:
        raise Exception(f'Unsetting hash set failed with status-code {response.status_code}...:'
                        + str(response.content))


def use_model_server(database_type, database_name, method, args, kwargs):
    if database_name not in databases:
        database = get_database_from_database_type(database_type, database_name)
        databases[database_name] = database
    bson_ = {
        'database': database_name,
        'method': method,
        'args': args,
        'kwargs': {
            **kwargs,
            'remote': False,
        },
    }
    body = BSON.encode(bson_)
    response = requests.post(
        f'http://{cf["model_server"]["host"]}:{cf["model_server"]["port"]}/',
        data=body,
    )
    out = BSON.decode(response.content)
    return out

