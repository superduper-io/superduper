import inspect
from ray import serve

from bson import BSON
from fastapi import Request
from fastapi.responses import Response
from superduperdb.datalayer.base import get_database_from_database_type
from superduperdb.cluster.annotations import decode_args, decode_kwargs, decode_result


class Server:
    def __init__(self):
        self.databases = {}

    def __call__(self, request: Request):
        data = request.body()
        data = BSON.decode(data)
        database_type = data['database_type']
        database_name = data['database_name']
        if f'{database_name}' not in self.databases:
            self.databases[f'{database_name}'] = \
                get_database_from_database_type(database_type, database_name)
        database = self.databases[data['database']]
        method = getattr(database, data['method'])
        args = decode_args(database,
                           inspect.signature(method),
                           data['args'])
        kwargs = decode_kwargs(database,
                               inspect.signature(method),
                               data['kwargs'])
        result = method(*args, **kwargs, remote=False)
        result = decode_result(database, inspect.signature(method), result)
        return Response(content=result, media_type="application/octet-stream")


if __name__ == '__main__':
    serve.create_backend("ray_server", Server())
    serve.create_endpoint("apply_model", backend="ray_server", route="/apply_model")
