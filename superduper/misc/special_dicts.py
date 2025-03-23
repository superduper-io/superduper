import typing as t
from collections import OrderedDict

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree


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


# TODO - incorporate into `Document`
class DeepKeyedDict(t.Dict[str, t.Any]):
    """Dictionary object mirroring how MongoDB handles fields.

    :param args: *args for `dict`
    :param kwargs: **kwargs for `dict`
    """

    def items(self, deep: bool = False):
        """Returns an iterator of key-value pairs.

        :param deep: `bool` whether to iterate over all nested key-value pairs.
        """
        for key, value in super().items():
            if deep and isinstance(value, DeepKeyedDict):
                for sub_key, sub_value in value.items(deep=True):
                    yield f'{key}.{sub_key}', sub_value
            else:
                yield key, value

    def keys(self, deep: bool = False):
        """Returns an iterator of keys.

        :param deep: `bool` whether to iterate over all nested keys.
        """
        for key in super().keys():
            if deep and isinstance(self[key], DeepKeyedDict):
                for sub_key in self[key].keys(deep=True):
                    yield f'{key}.{sub_key}'
            else:
                yield key

    def __getitem__(self, key: str) -> t.Any:
        # TODO - handle numeric keys
        if key == '_base' and '_base' not in self:
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
                sub = DeepKeyedDict(self.__getitem__(parent))
                return sub[child]

    def __setitem__(self, key: str, value: t.Any) -> None:
        if '.' not in key:
            super().__setitem__(key, value)
        else:
            parent = key.split('.')[0]
            child = '.'.join(key.split('.')[1:])
            try:
                parent_item = DeepKeyedDict(self[parent])
            except KeyError:
                parent_item = DeepKeyedDict({})
            parent_item[child] = value
            self[parent] = parent_item


# TODO duplicate in document.py
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


# TODO: Use this function to replace the similar logic in codebase
def recursive_update(data, replace_function: t.Callable):
    """Recursively update data with a replace function.

    :param data: Dict, List, Tuple, Set
    :param replace_function: Callable
    """
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = recursive_update(value, replace_function)
        return data
    elif isinstance(data, (list, tuple, set)):
        updated = (recursive_update(item, replace_function) for item in data)
        return type(data)(updated)
    else:
        return replace_function(data)


# TODO: Use this function to replace the similar logic in codebase
def recursive_find(data, check_function: t.Callable):
    """Recursively find items in data that satisfy a check function.

    :param data: Dict, List, Tuple, Set
    :param check_function: Callable
    """
    found_items = []

    def recurse(data):
        if isinstance(data, dict):
            for value in data.values():
                recurse(value)
        elif isinstance(data, (list, tuple, set)):
            for item in data:
                recurse(item)
        else:
            if check_function(data):
                found_items.append(data)

    recurse(data)
    return found_items


def dict_to_ascii_table(d):
    """
    Return a single string that represents an ASCII table.

    Each key/value in the dict is a column.
    Columns are centered and padded based on the widest
    string needed (key or value).

    :param d: Convert a dictionary to a table.
    """
    if not d:
        return "<empty dictionary>"

    keys = list(d.keys())
    vals = list(d.values())

    # Determine the needed width for each column
    widths = [max(len(str(k)), len(str(v))) for k, v in zip(keys, vals)]

    def center_text(text, width):
        """Center text within a given width using spaces."""
        text = str(text)
        if len(text) >= width:
            return text  # already as wide or wider, won't cut off
        # Calculate left/right spaces for centering
        left_spaces = (width - len(text)) // 2
        right_spaces = width - len(text) - left_spaces
        return " " * left_spaces + text + " " * right_spaces

    # Build the header row (keys)
    header_row = " | ".join(center_text(k, w) for k, w in zip(keys, widths))

    # Build a separator row with + in the middle
    separator_row = "-+-".join("-" * w for w in widths)

    # Build the value row
    value_row = " | ".join(center_text(v, w) for v, w in zip(vals, widths))

    # Combine them with line breaks
    return "\n".join([header_row, separator_row, value_row])
