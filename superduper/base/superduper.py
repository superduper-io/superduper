import re
import typing as t

from superduper.base.configs import CFG

__all__ = ('superduper',)


def superduper(item: t.Optional[t.Any] = None, **kwargs) -> t.Any:
    """Superduper API to automatically wrap an object to a db or a component.

    Attempts to automatically wrap an item in a superduper.ioponent by
    using duck typing to recognize it.

    :param item: A database or model
    :param kwargs: Additional keyword arguments to pass to the component
    """
    if item is None:
        item = CFG.data_backend

    if isinstance(item, str):
        return _auto_identify_connection_string(item, **kwargs)

    return _DuckTyper.run(item, **kwargs)


def _auto_identify_connection_string(item: str, **kwargs) -> t.Any:
    from superduper.base.build import build_datalayer

    if item.startswith('mongomock://'):
        kwargs['data_backend'] = item

    elif item.startswith('mongodb://'):
        kwargs['data_backend'] = item

    elif item.startswith('mongodb+srv://') and 'mongodb.net' in item:
        kwargs['data_backend'] = item

    elif item.endswith('.csv'):
        if CFG.cluster.cdc.uri is not None:
            raise TypeError('Pandas is not supported in cluster mode!')
        kwargs['data_backend'] = item

    else:
        if re.match(r'^[a-zA-Z0-9]+://', item) is None:
            raise ValueError(f'{item} is not a valid connection string')
        kwargs['data_backend'] = item
    return build_datalayer(CFG, **kwargs)


class _DuckTyper:
    attrs: t.Sequence[str]
    count: int

    @staticmethod
    def run(item: t.Any, **kwargs) -> t.Any:
        """
        Run the DuckTyper on an item.

        :param item: The item to run the DuckTyper on.
        :param kwargs: Additional keyword arguments to pass to the Duck
        """
        dts = [dt for dt in _DuckTyper._DUCK_TYPES if dt.accept(item)]
        if not dts:
            raise ValueError(
                f'Couldn\'t auto-identify {item}, please wrap explicitly using '
                '``superduper.components.*``'
            )

        if len(dts) == 1:
            return dts[0].create(item, **kwargs)

        raise ValueError(f'{item} matched more than one type: {dts}')

    # TODO: Does this item match the DuckType?
    @classmethod
    def accept(cls, item: t.Any) -> bool:
        """Check if an item matches the DuckType.

        The default implementation returns True if the number of attrs that
        the item has is exactly equal to self.count.

        """
        return sum(hasattr(item, a) for a in cls.attrs) == cls.count

    @classmethod
    def create(cls, item: t.Any, **kwargs) -> t.Any:
        """Create a component from the item.

        This method should be implemented by subclasses.
        :param item: The item to create the component from.
        """
        raise NotImplementedError

    _DUCK_TYPES: t.List[t.Type] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _DuckTyper._DUCK_TYPES.append(cls)


class MongoDbTyper(_DuckTyper):
    """A DuckTyper for MongoDB databases.

    This DuckTyper is used to automatically wrap a MongoDB database in a
    Datalayer.   # noqa
    """

    attrs = ('list_collection_names',)
    count = len(attrs)

    @classmethod
    def accept(cls, item: t.Any) -> bool:
        """Check if an item is a MongoDB database.

        :param item: The item to check.
        """
        return super().accept(item) and item.__class__.__name__ == 'Database'

    @classmethod
    def create(cls, item: t.Any, **kwargs) -> t.Any:
        """Create a Datalayer from a MongoDB database.

        :param item: A MongoDB database.
        """
        from mongomock.database import Database as MockDatabase
        from pymongo.database import Database

        from superduper import logging
        from superduper.backends.mongodb.data_backend import MongoDataBackend
        from superduper.base.build import build_datalayer

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
    """A DuckTyper for scikit-learn estimators # noqa.

    This DuckTyper is used to automatically wrap a scikit-learn estimator in
    an Estimator.
    """

    attrs = '_predict', 'fit', 'score', 'transform'
    count = 2

    @classmethod
    def create(cls, item: t.Any, **kwargs) -> t.Any:
        """Create an Estimator from a scikit-learn estimator.

        :param item: A scikit-learn estimator.
        """
        from sklearn.base import BaseEstimator

        from superduper.ext.sklearn.model import Estimator

        if not isinstance(item, BaseEstimator):
            raise TypeError('Expected BaseEstimator but got {type(item)}')

        kwargs['identifier'] = _auto_identify(item)
        return Estimator(object=item, **kwargs)


class TorchTyper(_DuckTyper):
    """A DuckTyper for torch.nn.Module and torch.jit.ScriptModule.

    This DuckTyper is used to automatically wrap a torch.nn.Module or
    torch.jit.ScriptModule in a TorchModel. # noqa
    """

    attrs = 'forward', 'parameters', 'state_dict', '_load_from_state_dict'
    count = len(attrs)

    @classmethod
    def create(cls, item: t.Any, **kwargs) -> t.Any:
        """Create a TorchModel from a torch.nn.Module or torch.jit.ScriptModule.

        :param item: A torch.nn.Module or torch.jit.ScriptModule.
        """
        from torch import jit, nn

        from superduper.ext.torch.model import TorchModel

        if isinstance(item, nn.Module) or isinstance(item, jit.ScriptModule):
            return TorchModel(identifier=_auto_identify(item), object=item, **kwargs)

        raise TypeError(f'Expected a Module but got {type(item)}')


def _auto_identify(instance: t.Any) -> str:
    return instance.__class__.__name__.lower()
