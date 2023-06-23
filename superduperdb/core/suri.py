from .documents import Document
from superduperdb.misc.key_cache import Cache, KeyCache
import abc
import superduperdb as s
import typing as t

T = t.TypeVar('T')


class SURI(s.JSONable, t.Generic[T], abc.ABC):
    uri: str = ''
    _cache: t.ClassVar[Cache]

    def __call__(self) -> T:
        return self._cache.get(self.uri)

    @classmethod
    def add(cls, item: T) -> t.Any:
        """Add a Document to the cache and return the SURI"""
        return cls(uri=cls._cache.put(item))


class URIDocument(SURI[Document]):
    _cache: t.ClassVar[KeyCache[Document]] = KeyCache[Document]()
