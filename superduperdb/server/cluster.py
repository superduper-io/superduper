import uvicorn


def _superduper_local_service(name, app, cfg=None):
    from superduperdb.server.app import Server

    service = getattr(cfg.cluster, name)
    assert isinstance(service, str)
    port = int(service.split(':')[-1])

    app.pre_start(cfg)
    config = uvicorn.Config(app.app, port=port, log_level="info")
    return Server(config=config)


class _Client:
    def __init__(self, db, services=()):
        self.db = db
        self.services = services
        for service in services:
            service.run_in_thread()

    def close(self):
        self.db.compute.shutdown()
        for service in self.services:
            service.stop()


def local_cluster(db):
    """
    This method is used to create a local cluster consisting of
    Vector search service, cdc service and dask setup.

    Once this cluster is up, user can offload vector search,
    cdc on these services.
    """
    from superduperdb import CFG

    # vector search local service
    CFG.force_set('cluster.vector_search', 'http://localhost:8000')
    CFG.force_set('cluster.cdc', 'http://localhost:8001')
    CFG.force_set('cluster.compute', 'dask+thread')

    from superduperdb.vector_search.server.app import app as vector_search_app

    vector_search_server = _superduper_local_service(
        'vector_search', vector_search_app, CFG
    )

    # cdc local service
    from superduperdb.cdc.app import app as cdc_app

    cdc_server = _superduper_local_service('cdc', cdc_app, CFG)

    # local compute
    from superduperdb.backends.dask.compute import DaskComputeBackend

    local_compute = DaskComputeBackend('', local=True)
    db.set_compute(local_compute)

    return _Client(db=db, services=(cdc_server, vector_search_server))
