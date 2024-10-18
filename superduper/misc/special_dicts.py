import copy
import re
import typing as t
from collections import OrderedDict

import yaml
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from superduper.base.constant import KEY_BLOBS, KEY_BUILDS, KEY_FILES
from superduper.base.variables import _find_variables

if t.TYPE_CHECKING:
    pass


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


def _highlight_references(yaml_str, pattern=r'(\?[\w/<>:.-]+|\&[\w/<>:.-]+)'):
    highlighted_text = Text()
    for line in yaml_str.split('\n'):
        matches = re.finditer(pattern, line)
        start = 0
        for match in matches:
            before = line[start : match.start()]
            reference = line[match.start() : match.end()]
            start = match.end()
            if reference.startswith('?'):
                highlighted_text.append(before)
                highlighted_text.append(reference, style="bold underline green")
            elif reference.startswith('&'):
                highlighted_text.append(before)
                highlighted_text.append(reference, style="bold underline blue")

        if start < len(line):
            after = line[start:]
            field = after.split(':')
            if len(field) > 1:
                highlighted_text.append(field[0] + ':', style="bold magenta")
                highlighted_text.append(':'.join(field[1:]))
            else:
                highlighted_text.append(after)
        highlighted_text.append('\n')
    return highlighted_text


def _format_base_section(base_section):
    blocks = base_section.split('/')
    component_type = blocks[1]
    component_identifier = blocks[-1]
    base = {
        'type': component_type,
        'identifier': component_identifier,
        'reference': base_section,
    }

    yaml_str = yaml.dump(base)
    return yaml_str


def _print_serialized_object(serialized_object):
    console = Console()

    # Extract sections of the serialized object
    base_section = serialized_object.get('_base', {})
    leaves_section = serialized_object.get(KEY_BUILDS, {})
    blobs_section = serialized_object.get(KEY_BLOBS, {})

    # Format base section with additional component info
    base_yaml = _format_base_section(base_section)

    # Convert sections to YAML strings
    leaves_yaml = yaml.dump(leaves_section)
    blobs_yaml = yaml.dump(blobs_section)

    # Highlight references
    base_text = _highlight_references(base_yaml)
    leaves_text = _highlight_references(leaves_yaml)
    blobs_text = _highlight_references(blobs_yaml)

    # Create panels for different sections
    base_panel = Panel(base_text, title="Base Component", border_style="green")
    leaves_panel = Panel(leaves_text, title="Leaves", border_style="cyan")
    blobs_panel = Panel(blobs_text, title="Blobs (Binary data)", border_style="magenta")

    # Print the panels
    console.print(base_panel)
    console.print(leaves_panel)
    console.print(blobs_panel)
    console.print(_legend_panel())


def _legend_panel():
    legend_content = """
    [bold underline green]Legend[/]\n
    [bold blue]_path[/]: This indicates the module path within superduper.io.\n
    [bold yellow]&[/]: This denotes an artifact store reference.\n
    [bold magenta]?[/]: This represents an intra-serialized data reference, 
                        typically within the child leaves.\n
    """
    return Panel(legend_content, title="Legend", border_style="green", width=50)


class SuperDuperFlatEncode(t.Dict[str, t.Any]):
    """
    Dictionary for representing flattened encoding data.

    :param args: *args for `dict`
    :param kwargs: **kwargs for `dict`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def builds(self):
        """Return the builds of the dictionary."""
        return IndexableDict(self.get(KEY_BUILDS, {}))

    @staticmethod
    def _str2var(x, item, variable):
        if isinstance(x, str):
            return x.replace(item, f'<var:{variable}>')
        if isinstance(x, dict):
            return {
                SuperDuperFlatEncode._str2var(
                    k, item, variable
                ): SuperDuperFlatEncode._str2var(v, item, variable)
                for k, v in x.items()
            }
        if isinstance(x, list):
            return [SuperDuperFlatEncode._str2var(v, item, variable) for v in x]
        return x

    def create_template(self, **kwargs):
        """Convert all instances of string to variable."""
        r = self
        for k, v in kwargs.items():
            r = SuperDuperFlatEncode._str2var(r, v, k)
        r['_variables'] = {v: f'<value-{i}>' for i, v in enumerate(kwargs.values())}
        return SuperDuperFlatEncode(r)

    @property
    def files(self):
        """Return the files of the dictionary."""
        return self.get(KEY_FILES, [])

    @property
    def blobs(self):
        """Return the blobs of the dictionary."""
        return self.get(KEY_BLOBS, [])

    def pop_builds(self):
        """Pop the builds of the dictionary."""
        return IndexableDict(self.pop(KEY_BUILDS, {}))

    def pop_files(self):
        """Pop the files of the dictionary."""
        return self.pop(KEY_FILES, {})

    def pop_blobs(self):
        """Pop the blobs of the dictionary."""
        return self.pop(KEY_BLOBS, {})

    def load_keys_with_blob(self):
        """Load all outer reference keys with actual data blob."""

        def _get_blob(output, key):
            if isinstance(key, str) and key[0] == '?':
                output = output[KEY_BUILDS][key[1:]]['blob']
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
        builds = copy.deepcopy(self.builds)
        builds.update(d.builds)

        blobs = copy.deepcopy(self.blobs)
        blobs = blobs.append(d.blobs)

        files = copy.deepcopy(self.files)
        files = files.append(d.files)

        if inplace:
            if builds:
                self[KEY_BUILDS] = builds
            self[KEY_BLOBS] = blobs
            self[KEY_FILES] = files
            return self
        else:
            out = copy.deepcopy(self)
            if builds:
                out[KEY_BUILDS] = builds
            out[KEY_BLOBS] = blobs
            out[KEY_FILES] = files
            return out

    @property
    def variables(self):
        """List of variables in the object."""
        return sorted(list(set(_find_variables(self))))

    def info(self):
        """Print the serialized object."""
        _print_serialized_object(self)


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


class ArgumentDefaultDict(dict):
    """
    Default-dictionary which takes the key as an argument to default factory.

    :param args: *args for `dict`
    :param default_factory: Callable used to create default dependent on key
    :param kwargs: **kwargs for `dict`
    """

    def __init__(self, *args, default_factory, **kwargs):
        self.default_factory = default_factory
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key not in self:
            self[key] = self.default_factory(key)
        return super().__getitem__(key)
