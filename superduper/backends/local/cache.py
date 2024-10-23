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
        self._component_to_uuid: t.Dict = {}
        self._db = None

    def list_components(self):
        """List components by (type_id, identifier) in the cache."""
        return list(self._component_to_uuid.keys())

    def list_uuids(self):
        """List UUIDs in the cache."""
        return list(self._cache.keys())

    def __getitem__(self, item):
        if isinstance(item, tuple):
            # (type_id, identifier)
            item = self._component_to_uuid[item[0], item[1]]
        return self._cache[item]

    def _put(self, component: Component):
        """Put a component in the cache."""
        self._cache[component.uuid] = component
        if (component.type_id, component.identifier) in self._component_to_uuid:
            current = self._component_to_uuid[component.type_id, component.identifier]
            current_version = self._cache[current].version
            if current_version < component.version:
                self._component_to_uuid[
                    component.type_id, component.identifier
                ] = component.uuid
        else:
            self._component_to_uuid[
                component.type_id, component.identifier
            ] = component.uuid

    def __delitem__(self, item):
        if isinstance(item, tuple):
            item = self._component_to_uuid[item[0], item[1]]
        tuples = [k for k, v in self._component_to_uuid.items() if v == item]
        if tuples:
            for type_id, identifier in tuples:
                del self._component_to_uuid[type_id, identifier]
        del self._cache[item]

    def initialize(self):
        """Initialize the cache."""

    def drop(self):
        """Drop the cache."""
        self._cache = {}
        self._component_to_uuid = {}

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

    def __iter__(self):
        return iter(self._cache.keys())

    def expire(self, item):
        """Expire an item from the cache."""
        try:
            del self._cache[item]
            for (t, i), uuid in self._component_to_uuid.items():
                if uuid == item:
                    del self._component_to_uuid[t, i]
                    break
        except KeyError:
            pass
