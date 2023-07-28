import abc
import dataclasses as dc
import time
import typing as t
from threading import Lock

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
                key = str(self._count)
                self._count += 1
            elif key in self:
                raise ValueError(f'Tried to insert duplicate key {key}')

            self._cache[key] = entry, time.time()
            self._inverse[entry] = key

            return key

    def get(self, key: str) -> Entry:
        """Given a key, returns an entry or raises KeyError"""
        with self._lock:
            return self._cache[key][0]

    def expire(self, before: float) -> t.Dict[str, t.Any]:
        with self._lock:
            old = {k: e for k, (e, time) in self._cache.items() if time < before}
            for key, entry in old.items():
                self._cache.pop(key)
                self._inverse.pop(entry)
            return old

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    _cache: t.Dict[str, t.Tuple[Entry, float]] = dc.field(default_factory=dict)
    _count: int = 0
    _inverse: t.Dict[Entry, str] = dc.field(default_factory=dict)
    _lock: Lock = dc.field(default_factory=Lock)
