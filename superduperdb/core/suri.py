from .documents import Document
from overrides import override
from superduperdb.misc.jsonable import Box
from superduperdb.misc.key_cache import Cache, KeyCache
import abc
import typing as t

Contents = t.TypeVar('Contents')


class SURI(Box[Contents], abc.ABC):
    uri: str = ''
    _cache: t.ClassVar[Cache]

    @classmethod
    def box(cls, contents: Contents) -> 'SURI[Contents]':
        return super().box(contents)  # type: ignore[return-value]

    @override
    def _box_to_contents(self) -> Contents:
        return self._cache.get(self.uri)

    @classmethod
    def _contents_to_box(cls, contents: Contents) -> 'SURI[Contents]':
        return cls(uri=cls._cache.put(contents))


class URIDocument(SURI[Document]):
    _cache: t.ClassVar[KeyCache[Document]] = KeyCache[Document]()
