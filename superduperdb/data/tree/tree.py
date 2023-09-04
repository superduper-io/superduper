import typing as t

"""
Utilities to deal with trees made up recursively of dicts and lists,
with leaves of any type.
"""

Accept = t.Callable[[t.Any], bool]
Rewrite = t.Callable[[t.Any], t.Any]


def tree_find(tree: t.Any, accept: Accept) -> t.Iterator[t.Any]:
    """Iterates recursively through lists and dicts, yielding all leaves `x`
       where `accept(x)` is True.

    :param tree: A tree made of dicts and lists
    :param accept: A function which returns True for values to yield
    """
    if accept(tree):
        yield tree
        return

    if isinstance(tree, dict):
        it = iter(tree.values())
    elif isinstance(tree, list):
        it = iter(tree)
    else:
        return

    yield from (a for i in it for a in tree_find(i, accept))


def tree_rewrite(tree: t.Any, accept: Accept, rewrite: Rewrite) -> t.Any:
    """Iterates recursively through lists and dicts, rewriting all leaves `x`
       where `accept(x)` is True with `rewrite(x)`

    :param tree: A tree made of dicts and lists
    :param accept: A function which returns True for values to rewrite
    :param rewrite: A function which rewrites accepted values
    """
    if accept(tree):
        return rewrite(tree)
    if isinstance(tree, list):
        return [tree_rewrite(t, accept, rewrite) for t in tree]
    if isinstance(tree, dict):
        return {k: tree_rewrite(v, accept, rewrite) for k, v in tree.items()}
    return tree
