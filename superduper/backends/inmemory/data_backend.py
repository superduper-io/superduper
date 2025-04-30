import json
import os
import typing as t
from fnmatch import fnmatch

import click

from superduper.backends.base.data_backend import KeyedDatabackend


class InMemoryDatabackend(KeyedDatabackend):
    """In-memory databackend for testing purposes.

    :param uri: URI to the in-memory database.
    :param plugin: Plugin implementing the in-memory database.
    :param flavour: Flavour of the in-memory database.
    """

    def __init__(self, uri: str, *args, **kwargs):
        super().__init__(uri, *args, **kwargs)

        self.path = uri.split('inmemory://')[1]

        if self.path:
            try:
                with open(self.path, 'r') as f:
                    self.data = json.load(f)
            except FileNotFoundError:
                self.data = {}
        else:
            self.path = None
            self.data = {}

    def insert(self, *args, **kwargs):
        out = super().insert(*args, **kwargs)
        if self.path:
            with open(self.path, 'w') as f:
                json.dump(self.data, f)
        return out

    def update(self, *args, **kwargs):
        out = super().update(*args, **kwargs)
        if self.path:
            with open(self.path, 'w') as f:
                json.dump(self.data, f)
        return out

    def delete(self, *args, **kwargs):
        out = super().delete(*args, **kwargs)
        if self.path:
            with open(self.path, 'w') as f:
                json.dump(self.data, f)
        return out

    def drop(self, force: bool = False):
        """Drop the in-memory store.

        :param force: Force drop the in-memory store.
        """
        if not force and not click.confirm(
            'Are you sure you want to drop the in-memory store?'
            ' This will delete all data in the store.',
            default=False,
        ):
            return
        if self.path:
            os.remove(self.path)
        self.data = {}

    def keys(self, *pattern: str):
        """Get the keys in the in-memory store.

        :param pattern: Pattern to match the keys.
        """
        all = ['/'.join(x) for x in self.data.keys()]
        pattern = '/'.join(pattern)
        if pattern:
            all = [
                tuple(x.split('/')) for x in all if fnmatch(x, pattern)  # type: ignore[type-var]
            ]
        else:
            all = [tuple(x.split('/')) for x in all]
        return all

    def reconnect(self):
        """Reconnect to the in-memory store."""
        pass

    def __delitem__(self, key):
        del self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data.get(key)
