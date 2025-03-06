import copy
import fnmatch
import re
import typing as t

from superduper.backends.base.cache import Cache
from superduper.components.component import Component


class LocalCache(Cache):
    """Local cache for caching components.

    :param init_cache: Initialize cache
    """

    def __init__(self):
        self._cache: t.Dict = {}

    def __delitem__(self, item):
        del self._cache[item]

    def __setitem__(self, key, value):
        self._cache[key] = copy.deepcopy(value)

    def __getitem__(self, item):
        out = self._cache[item]
        return copy.deepcopy(out)

    def __contains__(self, key):
        return key in self._cache

    def keys(self, *pattern):
        if not pattern:
            return list(self._cache.keys())
        else:
            pattern = ':'.join(pattern)
        keys = [':'.join(x) for x in self._cache.keys()]
        matched = fnmatch.filter(list(keys), pattern)
        return [tuple(x.split(':')) for x in matched]

    def initialize(self):
        """Initialize the cache."""
        pass

    def drop(self, force: bool = False):
        """Drop component from the cache.

        :param uuid: Component uuid.
        """
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
