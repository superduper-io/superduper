import copy
import typing as t
from collections import OrderedDict, defaultdict


class IndexableDict(OrderedDict):
    """IndexableDict.

    Example:
    -------
    >>> d = IndexableDict({'a': 1, 'b': 2})
    >>> d[0]
    1

    >>> d[1]
    2

    :param ordered_dict: OrderedDict

    """

    def __init__(self, ordered_dict):
        self._ordered_dict = ordered_dict
        self._keys = list(ordered_dict.keys())
        super().__init__(ordered_dict)

    def __getitem__(self, index):
        if isinstance(index, str):
            return super().__getitem__(index)
        try:
            return self[self._keys[index]]
        except IndexError:
            raise IndexError(f"Index {index} is out of range.")

    def __setitem__(self, index, value):
        if isinstance(index, str):
            super().__setitem__(index, value)
            return
        try:
            self[self._keys[index]] = value
        except IndexError:
            raise IndexError(f"Index {index} is out of range.")


class SuperDuperFlatEncode(t.Dict[str, t.Any]):
    """Dictionary for representing flattened encoding data."""

    @property
    def leaves(self):
        """Return the leaves of the dictionary."""
        return IndexableDict(self.get('_leaves', {}))

    @property
    def files(self):
        """Return the files of the dictionary."""
        return self.get('_files', [])

    @property
    def blobs(self):
        """Return the blobs of the dictionary."""
        return self.get('_blobs', [])

    def pop_leaves(self):
        """Pop the leaves of the dictionary."""
        return IndexableDict(self.pop('_leaves', {}))

    def pop_files(self):
        """Pop the files of the dictionary."""
        return self.pop('_files', [])

    def pop_blobs(self):
        """Pop the blobs of the dictionary."""
        return self.pop('_blobs', [])

    def load_keys_with_blob(self):
        """
        Load all outer reference keys with actual data
        blob.
        """

        def _get_blob(output, key):
            if isinstance(key, str) and key[0] == '?':
                output = output['_leaves'][key[1:]]['blob']
            else:
                output = key
            return output
        if '_base' in self:
            key = self['_base']
            return _get_blob(self, key)
        else:
            for k, v in self.items():
                self[k] = _get_blob(self, key=v) 
        return self

    def merge(self, d, inplace=False):
        """Merge two dictionaries.

        :param d: Dict, must have '_base' key
        :param inplace: bool, if True, merge in place
        """
        assert isinstance(d, SuperDuperFlatEncode)
        if '_base' in d:
            assert '_base' in self, "Cannot merge differently encoded data"
        leaves = copy.deepcopy(self.leaves)
        leaves.update(d.leaves)

        blobs = copy.deepcopy(self.blobs)
        blobs = blobs.append(d.blobs)

        files = copy.deepcopy(self.files)
        files = files.append(d.files)

        if inplace:
            if leaves:
                self['_leaves'] = leaves
            self['_blobs'] = blobs
            self['_files'] = files
            return self
        else:
            out = copy.deepcopy(self)
            if leaves:
                out['_leaves'] = leaves
            out['_blobs'] = blobs
            out['_files'] = files
            return out


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
        if key == '_base':
            return self
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
            try:
                parent_item = MongoStyleDict(self[parent])
            except KeyError:
                parent_item = MongoStyleDict({})
            parent_item[child] = value
            self[parent] = parent_item


# TODO: Is this an unused class?
class ArgumentDefaultDict(defaultdict):
    """ArgumentDefaultDict."""

    def __getitem__(self, item):
        if item not in self:
            self[item] = self.default_factory(item)
        return super().__getitem__(item)


def diff(r1, r2):
    """Get the difference between two dictionaries.

    >>> _diff({'a': 1, 'b': 2}, {'a': 2, 'b': 2})
    {'a': (1, 2)}
    >>> _diff({'a': {'c': 3}, 'b': 2}, {'a': 2, 'b': 2})
    {'a': ({'c': 3}, 2)}

    :param r1: Dict
    :param r2: Dict
    """
    d = _diff_impl(r1, r2)
    out = {}
    for path, left, right in d:
        out['.'.join(path)] = (left, right)
    return out


def _diff_impl(r1, r2):
    if not isinstance(r1, dict) or not isinstance(r2, dict):
        if r1 == r2:
            return []
        return [([], r1, r2)]
    out = []
    for k in list(r1.keys()) + list(r2.keys()):
        if k not in r1:
            out.append(([k], None, r2[k]))
            continue
        if k not in r2:
            out.append(([k], r1[k], None))
            continue
        out.extend([([k, *x[0]], x[1], x[2]) for x in _diff_impl(r1[k], r2[k])])
    return out
