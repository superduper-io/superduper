from .documents import Document
from overrides import override
from pymongo.cursor import Cursor
from superduperdb.misc.jsonable import Box
from superduperdb.datalayer.base.cursor import SuperDuperCursor
from superduperdb.misc.key_cache import Cache, KeyCache
import abc
import typing as t

Contents = t.TypeVar('Contents')


class SURI(Box[Contents], abc.ABC):
    uri: str = ''
    cache: t.ClassVar[Cache]

    @override
    def _box_to_contents(self) -> Contents:
        return self.cache.get(self.uri)

    @classmethod
    def _contents_to_box(cls, contents: Contents) -> 'SURI[Contents]':
        return cls(uri=cls.cache.put(contents))


class URIDocument(SURI[Document]):
    @classmethod
    def box(cls, contents: Document) -> 'URIDocument':
        return super().box(contents)  # type: ignore[return-value]

    cache: t.ClassVar[KeyCache[Document]] = KeyCache[Document]()


class URICursor(SURI[SuperDuperCursor]):
    @classmethod
    def box(cls, contents: SuperDuperCursor) -> 'URICursor':
        return super().box(contents)  # type: ignore[return-value]

    def __iter__(self):
        return self()

    def __next__(self):
        return next(self())

    @property
    def raw_cursor(self) -> Cursor:
        return self().raw_cursor

    cache: t.ClassVar[KeyCache[SuperDuperCursor]] = KeyCache[SuperDuperCursor]()
