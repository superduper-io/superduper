import json
import re

import redis
from superduper import logging
from superduper.backends.base.cache import Cache


class RedisCache(Cache):
    """Local cache for caching components.

    :param init_cache: Initialize cache
    """

    def __init__(self, uri: str = 'redis://localhost:6379/0'):
        logging.info('Using Redis cache')
        logging.info(f'Connecting to Redis cache at {uri}')
        self.redis = redis.Redis.from_url(uri, decode_responses=True)
        logging.info(f'Connecting to Redis cache at {uri}... DONE')

    def __delitem__(self, item):
        self.redis.delete(':'.join(item))

    def __setitem__(self, key, value):
        key = ':'.join(key)
        self.redis.set(key, json.dumps(value))

    def keys(self, *pattern):
        """Get keys from the cache.

        :param pattern: The pattern to search for.
        """
        pattern = ':'.join(pattern)
        strings = list(self.redis.keys(pattern))
        return [tuple(re.split(':', string)) for string in strings]

    def __getitem__(self, item):
        out = self.redis.get(':'.join(item))
        if out is None:
            raise KeyError(item)
        return json.loads(out)

    def initialize(self):
        """Initialize the cache."""
        pass

    def drop(self, force: bool = False):
        """Drop component from the cache.

        :param uuid: Component uuid.
        """
        self.redis.flushdb()

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
