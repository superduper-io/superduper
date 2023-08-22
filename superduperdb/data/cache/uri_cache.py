import dataclasses as dc
import typing as t

from superduperdb.data.tree.for_each import for_each

from .typed_cache import TypedCache

ContentType = t.TypeVar('ContentType')


@dc.dataclass
class Cached(t.Generic[ContentType]):
    """A base class for classes that are cached using their uri and type"""

    _content: dc.InitVar[ContentType] = None
    uri: str = ''

    @property
    def content(self) -> ContentType:
        """Get the cached content. `content` is not JSONized"""
        return self._content  # type: ignore[attr-defined]

    @content.setter
    def content(self, content: ContentType):
        """Set the cached content. `content` is not JSONized"""
        self._content = content  # type: ignore[attr-defined]

    def __post_init__(self, content: ContentType):
        self.content = content


class URICache(TypedCache):
    """
    A typed cache for instances of `Cached`.
    """

    def cache(self, entry: t.Any) -> None:
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
