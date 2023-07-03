from superduperdb.datalayer.base import database
from superduperdb.misc import uri_cache
from superduperdb.server.server import Server
import functools
import superduperdb as s
import typing as t


def make_server(db: t.Any, cfg: t.Optional[s.config.Server] = None) -> Server:
    cache = uri_cache.URICache()
    server = Server(cfg=cfg or s.CFG.server, document_store=cache)

    def add_endpoint(name):
        method = getattr(db, name)

        @functools.wraps(method)
        def wrapped_endpoint(*a, **ka):
            cache.uncache_all([a, ka])

            result = method(*a, **ka)

            cache.cache_all(result)
            return result

        wrapped_endpoint.__name__ = name
        server.register(wrapped_endpoint)

    [add_endpoint(i) for i in database.ENDPOINTS]
    server.add_endpoints(db)
    return server
