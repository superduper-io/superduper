from .for_each import for_each
from .typed_cache import TypedCache
from superduperdb.misc import dataclasses as dc
import typing as t

ContentType = t.TypeVar('ContentType')


@dc.dataclass
class Cached:
    """A base class for classes that are cached using their uri and type"""

    uri: str = ''

    @classmethod
    def erase_fields(cls):
        cls.__dataclass_fields__ = Cached.__dataclass_fields__

    def __post_init__(self):
        assert isinstance(self.url, str), 'uri must be a string'


class URICache(TypedCache):
    """
    A typed cache for instances of `Cached`.
    """

    def cache(self, entry):
        """Add a uri to the content in an instance of Cached"""
        if isinstance(entry, Cached):
            entry.uri = self.put(entry.content)

    def uncache(self, entry: t.Any) -> None:
        """Retrieve the content from the uri in an instance of Cached"""
        if isinstance(entry, Cached):
            entry.content = self.get(entry.uri)

    def cache_all(self, entry: t.Any) -> None:
        """Run self.cache recursively"""
        for_each(self.cache, entry)

    def uncache_all(self, entry: t.Any) -> None:
        """Run self.uncache recursively"""
        for_each(self.uncache, entry)
