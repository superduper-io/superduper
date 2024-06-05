import typing as t

from ray import serve

from superduperdb import CFG
from superduperdb.components.model import Model
from superduperdb.server.app import SuperDuperApp

# TODO: Try to get health endpoints from superduperdb app
app = SuperDuperApp()


def run(
    model: str,
    version: t.Optional[int] = None,
    num_replicas: int = 1,
    ray_actor_options: t.Dict = {},
    route_prefix: str = '/',
):
    """Serve a superduperdb model on ray cluster.

    :param model: model identifier
    :param version: model version
    :param num_replicas: number of replicas
    :param ray_actor_options: ray actor options
    :param route_prefix: route prefix
    """

    @serve.deployment(ray_actor_options=ray_actor_options, num_replicas=num_replicas)
    @serve.ingress(app.app)
    class SuperDuperRayServe:
        """A ray deployment which serves a superduperdb model with default ingress."""

        def __init__(self, model_identifier: str, version: t.Optional[int]):
            from superduperdb.base.build import build_datalayer

            db = build_datalayer(CFG)
            self.model = db.load('model', model_identifier, version=version)

        @app.app.post("/predict")
        def predict(self, args: t.List, kwargs: t.Dict):
            assert isinstance(self.model, Model)
            return self.model.predict(*args, **kwargs)

    serve.run(
        SuperDuperRayServe.bind(model_identifier=model, version=version),
        route_prefix=route_prefix,
    )
