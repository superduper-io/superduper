from threading import Lock
import dataclasses as dc
import typing as t

Entry = t.TypeVar('Entry')


@dc.dataclass
class KeyCache(t.Generic[Entry]):
    def put(self, entry: Entry) -> str:
        """Put an item into the cache, return a string key"""
        with self._lock:
            try:
                return self._inverse[entry]
            except KeyError:
                pass

            key = str(len(self._cache))
            self._inverse[entry] = key
            self._cache[key] = entry

            return key

    def get(self, key: str) -> Entry:
        """Given a key, returns an entry or raises KeyError"""
        return self._cache[key]  # Atomic operation, no lock needed.

    _cache: t.Dict[str, Entry] = dc.field(default_factory=dict)
    _inverse: t.Dict[Entry, str] = dc.field(default_factory=dict)
    _lock: Lock = dc.field(default_factory=Lock)
