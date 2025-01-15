import re
import typing as t

from superduper.base.configs import CFG

__all__ = ('superduper',)


def superduper(item: str | None = None, **kwargs) -> t.Any:
    """Build a superduper connection.

    :param item: URI of connection.
    :param kwargs: Additional parameters to building `Datalayer`
    """
    from superduper.base.build import build_datalayer

    if item is None:
        return build_datalayer(**kwargs)

    assert isinstance(item, str), f'item must be a string, not {type(item)}'
    if re.match(r'^[a-zA-Z0-9]+://', item) is None:
        raise ValueError(f'{item} is not a valid connection string')

    kwargs['data_backend'] = item
    return build_datalayer(CFG, **kwargs)
