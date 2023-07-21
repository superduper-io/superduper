import typing as t

"""
Utilities to deal with trees made up recursive of dicts and lists,
with leaves of any type.
"""


def tree_find(tree: t.Any, accept: t.Callable[[t.Any], bool]) -> t.Iterator[t.Any]:
    """Iterates recursively through lists and dicts, yielding all leaves `x`
       where `accept(x)` is True.

    :param accept: A function which returns True for values to keep
    :param tree: A tree made of dicts and lists
    """
    if accept(tree):
        yield tree
        return

    if isinstance(tree, dict):
        it = tree.values()
    elif isinstance(tree, list):
        it = tree
    else:
        return

    yield from (a for i in it for a in tree_find(i, accept))
