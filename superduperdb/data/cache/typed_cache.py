import dataclasses as dc
import typing as t
from threading import Lock

from .key_cache import KeyCache

SEP = '-'


@dc.dataclass
class TypedCache:
    """Cache objects by class.

    Each class of object is given a unique name and its own cache.

    The key for an object is that unique class name, joined with the object's key from
    its class cache.
    """

    def put(self, entry: t.Any, key: t.Optional[str] = None) -> str:
        """Put an item into the cache, return a string key"""
        name = self._get_name(type(entry))
        cache = self._name_to_cache[name]
        if key is None:
            key = cache.put(entry)
            return f'{name}{SEP}{key}'

        k, _, rest = key.partition(SEP)
        if k == name and rest:
            cache.put(entry, rest)
            return key

        raise ValueError(f'Bad key {key}, expected {name}-')

    def _get_name(self, cls: t.Any) -> str:
        with self._lock:
            try:
                return self._class_to_name[cls]
            except KeyError:
                name = cls.__name__
                if name in self._name_to_cache:
                    name = f'{cls.__module__}.{name}'
                cache = KeyCache[cls]()

                self._class_to_name[cls] = name
                self._name_to_cache[name] = cache
        return name

    def get(self, key: str) -> t.Any:
        """Given a key, returns an entry or raises KeyError"""
        name, key = key.split(SEP)
        return self._name_to_cache[name].get(key)

    def expire(self, before: float) -> t.Dict[t.Type, t.Dict[str, t.Any]]:
        cn = self._class_to_name.items()
        cc = ((cls, self._name_to_cache[name]) for cls, name in cn)
        return {cls: cache.expire(before) for cls, cache in cc}

    def __contains__(self, key: str) -> bool:
        k, _, rest = key.partition(SEP)
        return rest in self._name_to_cache.get(k, {})

    def __len__(self) -> int:
        with self._lock:
            return sum(len(i) for i in self._name_to_cache.values())

    _class_to_name: t.Dict[t.Type, str] = dc.field(default_factory=dict)
    _lock: Lock = dc.field(default_factory=Lock)
    _name_to_cache: t.Dict[str, KeyCache] = dc.field(default_factory=dict)
