import click
from bson import BSON
from ray import serve as _serve
from starlette.requests import Request

from superduperdb.core.documents import Document
from superduperdb.datalayer.base.build import build_datalayer


def create_server(model: str, num_replicas):
    @_serve.deployment(
        route_prefix=f'/predict/{model}',
        num_replicas=num_replicas,
    )
    class Server:
        def __init__(self):
            self.db = build_datalayer()
            self.model = self.db.models[model]

        async def __call__(self, http_request: Request) -> str:
            data = await http_request.body()
            data = BSON.decode(data)  # type: ignore[arg-type]
            print(data)
            data = [Document.decode(r, encoders=self.db.encoders) for r in data]
            X = Document.decode(data['X'], encoders=self.db.encoders)
            try:
                X = X.unpack()
            except Exception:
                X = [x.unpack() for x in X]
            result = self.model.predict(X)
            return BSON.encode({'output': result.encode()})  # type: ignore

    return Server


@click.command()  # type: ignore[arg-type]
@click.argument('model')
@click.option('--num_replicas', default=1)
def serve(model: str, num_replicas: int = 1):
    Server = create_server(model, num_replicas=num_replicas)
    Server.bind()


if __name__ == '__main__':
    serve()
