from collections import defaultdict


class ArgumentDefaultDict(defaultdict):
    def __getitem__(self, item):
        if item not in self.keys():
            self[item] = self.default_factory(item)
        return super().__getitem__(item)


class MongoStyleDict(dict):
    """
    Dictionary object mirroring how fields can be referred to and set in MongoDB.

    >>> d = MongoStyleDict({'a': {'b': 1}})
    >>> d['a.b']
    1

    Set deep fields directly with string keys:
    >>> d['a.c'] = 2
    >>> d
    {'a': {'b': 1, 'c': 2}}

    Parent keys should exist in order to set subfields:
    >>> d['a.d.e'] = 3
    Traceback (most recent call last):
    ...
    KeyError: 'd'
    """
    def __getitem__(self, item):
        if '.' not in item:
            return super().__getitem__(item)
        parts = item.split('.')
        parent = parts[0]
        child = '.'.join(parts[1:])
        sub = MongoStyleDict(self.__getitem__(parent))
        return sub[child]

    def __setitem__(self, key, value):
        if '.' not in key:
            super().__setitem__(key, value)
        else:
            parent = key.split('.')[0]
            child = '.'.join(key.split('.')[1:])
            parent_item = MongoStyleDict(self[parent])
            parent_item[child] = value
            self[parent] = parent_item


class ExtensibleDict(defaultdict):
    def append(self, other):
        for k in other:
            self[k].append(other[k])