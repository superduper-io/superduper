import json

import click
import redis
from superduper import logging
from superduper.backends.base.data_backend import KeyedDatabackend


class RedisDataBackend(KeyedDatabackend):
    """Redis data backend for SuperDuper.

    :param uri: The Redis URI.
    :param plugin: The plugin instance.
    :param flavour: The flavour of the backend.
    """

    def __init__(self, uri: str, *args, **kwargs):
        super().__init__(uri, *args, **kwargs)
        self.reconnect()

    def drop(self, force: bool = False):
        """Drop the in-memory store.

        :param force: Force drop the in-memory store.
        """
        if not force and not click.confirm(
            'Are you sure you want to drop the in-memory store?'
        ):
            logging.warning('Aborting drop of in-memory store.')
            return
        self.conn.flushdb()

    def keys(self, *pattern: str):
        """Get the keys in the in-memory store.

        :param pattern: Pattern to match the keys.
        """
        matches = self.conn.keys('/'.join(pattern))
        return [tuple(x.split('/')) for x in matches]

    def reconnect(self):
        """Reconnect to the in-memory store."""
        self.conn = redis.from_url(
            self.uri,
            decode_responses=True,
        )

    def __delitem__(self, key):
        self.conn.delete('/'.join(key))

    def __setitem__(self, key, value):
        self.conn.set('/'.join(key), json.dumps(value))

    def __getitem__(self, key):
        return json.loads(self.conn.get('/'.join(key)))
