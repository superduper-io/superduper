import json
import subprocess
import typing as t

from . import command


@command(help='Start local dask scheduler')
def dask_scheduler():
    subprocess.Popen(["dask", "scheduler"])


@command(help='Start local dask worker')
def dask_worker():
    subprocess.Popen(["dask", "worker", "tcp://localhost:8786"])


@command(help='Start local cluster: server, dask and change data capture')
def local_cluster():
    from superduperdb.base.build import build_datalayer
    from superduperdb.server.cluster import cluster

    db = build_datalayer()
    cluster(db)


@command(help='Start vector search server')
def vector_search():
    from superduperdb.vector_search.server.app import app

    app.start()


@command(help='Start standalone change data capture')
def cdc():
    from superduperdb.cdc.app import app

    app.start()


@command(help='Serve a model on ray')
def ray_serve(
    model: str,
    version: t.Optional[int] = None,
    ray_actor_options: str = '',
    num_replicas: int = 1,
):
    from superduperdb.backends.ray.serve import run

    run(
        model=model,
        version=version,
        ray_actor_options=json.loads(ray_actor_options),
        num_replicas=num_replicas,
    )
