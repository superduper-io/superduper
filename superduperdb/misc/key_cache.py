from threading import Lock
import abc
import dataclasses as dc
import typing as t

Entry = t.TypeVar('Entry')


class Cache(abc.ABC):
    def put(self, entry: t.Any) -> str:
        raise NotImplementedError

    def get(self, key: str) -> t.Any:
        raise NotImplementedError


@dc.dataclass
class KeyCache(t.Generic[Entry], Cache):
    def put(self, entry: Entry, key: t.Optional[str] = None) -> str:
        """Put an item into the cache, return a string key"""
        with self._lock:
            try:
                return self._inverse[entry]
            except KeyError:
                pass

            if key is None:
                key = str(len(self._cache))
                while key in self:
                    key += 'x'  # Will rarely happen
            else:
                assert key not in self

            self._cache[key] = entry
            self._inverse[entry] = key

            return key

    def get(self, key: str) -> Entry:
        """Given a key, returns an entry or raises KeyError"""
        return self._cache[key]  # Atomic operation, no lock needed.

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    _cache: t.Dict[str, Entry] = dc.field(default_factory=dict)
    _inverse: t.Dict[Entry, str] = dc.field(default_factory=dict)
    _lock: Lock = dc.field(default_factory=Lock)
