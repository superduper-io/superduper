import re
import typing as t

from superduperdb.base.configs import CFG

__all__ = ('superduper',)


def superduper(item: t.Optional[t.Any] = None, **kwargs) -> t.Any:
    """
    Attempts to automatically wrap an item in a superduperdb component by
    using duck typing to recognize it.

    :param item: A database or model
    """

    if item is None:
        from superduperdb.base.build import build_datalayer

        return build_datalayer()

    if isinstance(item, str):
        return _auto_identify_connection_string(item, **kwargs)

    return _DuckTyper.run(item, **kwargs)


def _auto_identify_connection_string(item: str, **kwargs) -> t.Any:
    from superduperdb.base.build import build_datalayer

    if item.startswith('mongomock://'):
        CFG.force_set('data_backend', item)

    elif item.startswith('mongodb://'):
        CFG.force_set('data_backend', item)

    elif item.startswith('mongodb+srv://') and 'mongodb.net' in item:
        CFG.force_set('data_backend', item)
        CFG.force_set('vector_search', item)

    elif item.endswith('.csv'):
        CFG.force_set('data_backend', item)

    else:
        if re.match(r'^[a-zA-Z0-9]+://', item) is None:
            raise ValueError(f'{item} is not a valid connection string')
        CFG.force_set('data_backend', item)
    return build_datalayer(CFG, **kwargs)


class _DuckTyper:
    attrs: t.Sequence[str]
    count: int

    @staticmethod
    def run(item: t.Any, **kwargs) -> t.Any:
        dts = [dt for dt in _DuckTyper._DUCK_TYPES if dt.accept(item)]
        if not dts:
            raise ValueError(
                f'Couldn\'t auto-identify {item}, please wrap explicitly using '
                '``superduperdb.components.*``'
            )

        if len(dts) == 1:
            return dts[0].create(item, **kwargs)

        raise ValueError(f'{item} matched more than one type: {dts}')

    @classmethod
    def accept(cls, item: t.Any) -> bool:
        """Does this item match the DuckType?

        The default implementation returns True if the number of attrs that
        the item has is exactly equal to self.count.

        """
        return sum(hasattr(item, a) for a in cls.attrs) == cls.count

    @classmethod
    def create(cls, item: t.Any, **kwargs) -> t.Any:
        """Create a superduperdb component for an item that has already been accepted"""
        raise NotImplementedError

    _DUCK_TYPES: t.List[t.Type] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _DuckTyper._DUCK_TYPES.append(cls)


class MongoDbTyper(_DuckTyper):
    attrs = ('list_collection_names',)
    count = len(attrs)

    @classmethod
    def accept(cls, item: t.Any) -> bool:
        return super().accept(item) and item.__class__.__name__ == 'Database'

    @classmethod
    def create(cls, item: t.Any, **kwargs) -> t.Any:
        from mongomock.database import Database as MockDatabase
        from pymongo.database import Database

        from superduperdb import logging
        from superduperdb.backends.mongodb.data_backend import MongoDataBackend
        from superduperdb.base.build import build_datalayer

        if not isinstance(item, (Database, MockDatabase)):
            raise TypeError(f'Expected Database but got {type(item)}')

        logging.warn(
            'Note: This is only recommended in development mode, since config\
             still holds `data_backend` with the default value, services \
             like vector search and cdc cannot be reached due to configuration\
             mismatch. Services will be configured with a `data_backend` uri using \
             config file hence this client config and\
             services config will be different.'
        )
        databackend = MongoDataBackend(conn=item.client, name=item.name)
        return build_datalayer(cfg=CFG, databackend=databackend, **kwargs)


class SklearnTyper(_DuckTyper):
    attrs = '_predict', 'fit', 'score', 'transform'
    count = 2

    @classmethod
    def create(cls, item: t.Any, **kwargs) -> t.Any:
        from sklearn.base import BaseEstimator

        from superduperdb.ext.sklearn.model import Estimator

        if not isinstance(item, BaseEstimator):
            raise TypeError('Expected BaseEstimator but got {type(item)}')

        kwargs['identifier'] = _auto_identify(item)
        return Estimator(object=item, **kwargs)


class TorchTyper(_DuckTyper):
    attrs = 'forward', 'parameters', 'state_dict', '_load_from_state_dict'
    count = len(attrs)

    @classmethod
    def create(cls, item: t.Any, **kwargs) -> t.Any:
        from torch import jit, nn

        from superduperdb.ext.torch.model import TorchModel

        if isinstance(item, nn.Module) or isinstance(item, jit.ScriptModule):
            return TorchModel(identifier=_auto_identify(item), object=item, **kwargs)

        raise TypeError('Expected a Module but got {type(item)}')


def _auto_identify(instance: t.Any) -> str:
    return instance.__class__.__name__.lower()
