import dataclasses as dc
import typing as t

import superduperdb as s


def for_each(
    fn: t.Callable[[t.Any], t.Any], item: t.Any, depth_first: bool = False
) -> None:
    """
    Recursively applies a function, breadth-first.

    `for_each` recurses through dicts, lists and tuples, and also through
    members of dataclasses and Pydantic model.

    :param fn: function to apply
    :param item: item to apply function to
    :param depth_first: whether to apply the function before recursing
    """
    if not depth_first:
        fn(item)

    it: t.Iterable = ()

    if isinstance(item, (list, tuple)):
        it = item
    elif isinstance(item, dict):
        it = item.values()
    elif dc.is_dataclass(item):
        it = (getattr(item, f.name) for f in dc.fields(item))
    elif isinstance(item, s.JSONable):
        it = (getattr(item, i) for i in item.schema()['properties'])

    for i in it:
        for_each(fn, i)

    if depth_first:
        fn(item)
