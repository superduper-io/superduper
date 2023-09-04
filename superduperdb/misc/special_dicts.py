import typing as t


class MongoStyleDict(t.Dict[str, t.Any]):
    """
    Dictionary object mirroring how fields can be referred to and set in MongoDB.

    >>> d = MongoStyleDict({'a': {'b': 1}})
    >>> d['a.b']
    1

    t.Set deep fields directly with string keys:
    >>> d['a.c'] = 2
    >>> d
    {'a': {'b': 1, 'c': 2}}

    Parent keys should exist in order to set subfields:
    >>> d['a.d.e'] = 3
    Traceback (most recent call last):
    ...
    KeyError: 'd'
    """

    def __getitem__(self, key: str) -> t.Any:
        if '.' not in key:
            return super().__getitem__(key)
        else:
            try:
                return super().__getitem__(key)
            except KeyError:
                parts = key.split('.')
                parent = parts[0]
                child = '.'.join(parts[1:])
                sub = MongoStyleDict(self.__getitem__(parent))
                return sub[child]

    def __setitem__(self, key: str, value: t.Any) -> None:
        if '.' not in key:
            super().__setitem__(key, value)
        else:
            parent = key.split('.')[0]
            child = '.'.join(key.split('.')[1:])
            parent_item = MongoStyleDict(self[parent])
            parent_item[child] = value
            self[parent] = parent_item
