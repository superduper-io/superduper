import typing as t

from superduperdb.base.configs import CFG

__all__ = ('superduper',)


def superduper(item: t.Any, **kwargs) -> t.Any:
    """
    Attempts to automatically wrap an item in a superduperdb container by
    using duck typing to recognize it.

    :param item: A database or model
    """

    if isinstance(item, str):
        return _auto_identify_connection_string(item, **kwargs)

    return _DuckTyper.run(item, **kwargs)


def _auto_identify_connection_string(item: str, **kwargs) -> t.Any:
    from superduperdb.db.base.build import build_datalayer

    # cfg = copy.deepcopy(CFG)
    if item.startswith('mongomock://'):
        CFG.data_backend = item

    elif item.startswith('mongodb://'):
        CFG.data_backend = item

    elif item.startswith('mongodb+srv://') and 'mongodb.net' in item:
        CFG.data_backend = item
        CFG.vector_search = item

    else:
        raise NotImplementedError(f'Can\'t auto-identify connection string {item}')
    return build_datalayer(CFG)


class _DuckTyper:
    attrs: t.Sequence[str]
    count: int

    @staticmethod
    def run(item: t.Any, **kwargs) -> t.Any:
        dts = [dt for dt in _DuckTyper._DUCK_TYPES if dt.accept(item)]
        if not dts:
            raise NotImplementedError(
                f'Couldn\'t auto-identify {item}, please wrap explicitly using '
                '``superduperdb.container.*``'
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
        """Create a superduperdb container for an item that has already been accepted"""
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

        from superduperdb import CFG
        from superduperdb.db.base.build import build_vector_database
        from superduperdb.db.base.db import DB
        from superduperdb.db.mongodb.data_backend import MongoDataBackend

        if kwargs:
            raise ValueError('MongoDb creator accepts no parameters')
        if not isinstance(item, (Database, MockDatabase)):
            raise TypeError(f'Expected Database but got {type(item)}')

        databackend = MongoDataBackend(conn=item.client, name=item.name)
        return DB(
            databackend=databackend,
            metadata=databackend.build_metadata(),
            artifact_store=databackend.build_artifact_store(),
            vector_database=build_vector_database(CFG),
        )


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
