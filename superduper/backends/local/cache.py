import typing as t

from superduper.backends.base.cache import Cache
from superduper.components.component import Component


# TODO - doesn't need to be local, could work for services too
class LocalCache(Cache):
    """Local cache for caching components.

    :param init_cache: Initialize cache
    """

    def __init__(self, init_cache: bool = True):
        super().__init__()
        self.init_cache = init_cache
        self._cache: t.Dict = {}
        self._cache_uuid: t.Dict = {}
        self._db = None

    def list_components(self):
        """List components by (type_id, identifier) in the cache."""
        return list(self._cache.keys())

    def list_uuids(self):
        """List UUIDs in the cache."""
        return list(self._cache_uuid.keys())

    # TODO which of these is the correct one?
    # def __getitem__(self, *item):
    #     return self._cache[item]

    def _put(self, component: Component):
        """Put a component in the cache."""
        self._cache[component.type_id, component.identifier] = component
        self._cache_uuid[component.uuid] = component

    def __delitem__(self, name: str):
        del self._cache[name]

    def initialize(self):
        """Initialize the cache."""
        for component_data in self.db.show():
            type_id = component_data['type_id']
            identifier = component_data['identifier']
            r = self.db.show(type_id=type_id, identifier=identifier, version=-1)
            if r.get('cache', False):
                component = self.db.load(type_id=type_id, identifier=identifier)
                self.put(component)
                self.db.cluster.compute.put(component)

    def drop(self):
        """Drop the cache."""
        self._cache = {}

    @property
    def db(self):
        """Get the ``db``."""
        return self._db

    @db.setter
    def db(self, value):
        """Set the ``db``.

        :param value: The value to set the ``db`` to.
        """
        self._db = value
        self.initialize()

    # def init(self):
    #     """Initialize the cache."""
    #     if not self.init_cache:
    #         return
    #     for component in self.db.show():
    #         if 'version' not in component:
    #             component['version'] = -1

    #         show = self.db.show(**component)
    #         uuid = show.get('uuid')
    #         if show.get('cache', False):
    #             self._cache[uuid] = self.db.load(uuid=uuid)

    def __getitem__(self, uuid: str):
        """Get a component from the cache."""
        return self._cache_uuid[uuid]

    def __iter__(self):
        return iter(self._cache.keys())

    def expire(self, item):
        """Expire an item from the cache."""
        try:
            del self._cache[item]
        except KeyError:
            pass
