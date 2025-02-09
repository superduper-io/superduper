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
class MongoStyleDict(t.Dict[str, t.Any]):
    """Dictionary object mirroring how MongoDB handles fields.

    :param args: *args for `dict`
    :param kwargs: **kwargs for `dict`
    """

    def items(self, deep: bool = False):
        """Returns an iterator of key-value pairs.

        :param deep: `bool` whether to iterate over all nested key-value pairs.
        """
        for key, value in super().items():
            if deep and isinstance(value, MongoStyleDict):
                for sub_key, sub_value in value.items(deep=True):
                    yield f'{key}.{sub_key}', sub_value
            else:
                yield key, value

    def keys(self, deep: bool = False):
        """Returns an iterator of keys.

        :param deep: `bool` whether to iterate over all nested keys.
        """
        for key in super().keys():
            if deep and isinstance(self[key], MongoStyleDict):
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


def _childrens(tree, object, nesting=1):
    if not object.get('_builds', False) or not nesting:
        return
    for name, child in object.builds.items():
        identifier = child.get('uuid', None)
        child_text = f"{name}: {child.get('_path', '__main__')}({identifier})"
        subtree = tree.add(Text(child_text, style="yellow"))
        for key, value in child.items():
            key_text = Text(f"{key}", style="magenta")
            value_text = Text(f": {value}", style="blue")
            subtree.add(Text.assemble(key_text, value_text))
        _childrens(subtree, child, nesting=nesting - 1)


def _component_metadata(obj):
    metadata = []
    variables = obj.variables
    if variables:
        variable = "[yellow]Variables[/yellow]"
        metadata.append(variable)
        for var in variables:
            metadata.append(f"[magenta]{var}[/magenta]")
        metadata.append('\n')

    def _all_leaves(obj):
        result = []
        if not obj.leaves:
            return result
        for name, leaf in obj.leaves.items():
            tmp = _all_leaves(leaf)
            result.extend([f'{name}.{x}' for x in tmp])
            result.append(name)
        return result

    metadata.append("[yellow]Leaves[/yellow]")
    rleaves = _all_leaves(obj)
    obj_leaves = list(obj.leaves.keys())
    for leaf in rleaves:
        if leaf in obj_leaves:
            metadata.append(f"[green]{leaf}[/green]")
        else:
            metadata.append(f"[magenta]{leaf}[/magenta]")
    metadata = "\n".join(metadata)
    return metadata


def _display_component(obj, verbosity=1):
    from superduper.base.leaf import Leaf

    console = Console()

    MAX_STR_LENGTH = 50

    def _handle_list(lst):
        handled_list = []
        for item in lst:
            if isinstance(item, Leaf):
                if len(str(item)) > MAX_STR_LENGTH:
                    handled_list.append(str(item)[:MAX_STR_LENGTH] + "...")
                else:
                    handled_list.append(str(item))
            elif isinstance(item, (list, tuple)):
                handled_list.append(_handle_list(item))
            else:
                handled_list.append(str(item))
        return handled_list

    def _component_info(obj):
        base_component = []
        for key, value in obj.__dict__.items():
            if value is None:
                continue
            if isinstance(value, Leaf):
                if len(str(value)) > MAX_STR_LENGTH:
                    value = str(value)[:MAX_STR_LENGTH] + "..."
                else:
                    value = str(value)
            elif isinstance(value, (tuple, list)):
                value = _handle_list(value)
            base_component.append(f"[magenta]{key}[/magenta]: [blue]{value}[/blue]")

        base_component = "\n".join(base_component)
        return base_component

    base_component = _component_info(obj)
    base_component_metadata = _component_metadata(obj)

    properties_panel = Panel(
        base_component, title=obj.identifier, border_style="bold green"
    )

    if verbosity > 1:
        tree = Tree(
            Text(f'Component Map: {obj.identifier}', style="bold green"),
            guide_style="bold green",
        )
        _childrens(tree, obj.encode(), nesting=verbosity - 1)

    additional_info_panel = Panel(
        base_component_metadata, title="Component Metadata", border_style="blue"
    )
    panels = Columns([properties_panel, additional_info_panel])
    console.print(panels)
    if verbosity > 1:
        console.print(tree)


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
