import typing as t

__all__ = ('superduper',)


class DuckTyper:
    attrs: t.Sequence[str]
    count: int = 0

    @classmethod
    def run(cls, item: t.Any, **kwargs) -> t.Any:
        dts = [dt for dt in _DUCK_TYPES if dt.accept(item)]
        if len(dts) == 1:
            return dts[0].create(item, **kwargs)
        raise NotImplementedError(
            f'Couldn\'t auto-identify {item}, please wrap explicitly using '
            '``superduperdb.container.*``'
        )

    @classmethod
    def accept(cls, item: t.Any) -> bool:
        count = cls.count or len(cls.attrs)
        return sum(hasattr(item, a) for a in cls.attrs) == count

    @classmethod
    def create(cls, item: t.Any, **kwargs) -> t.Any:
        raise NotImplementedError


class MongoDbTyper(DuckTyper):
    attrs = ('list_collection_names',)

    @classmethod
    def accept(cls, item: t.Any) -> bool:
        count = cls.count or len(cls.attrs)
        test_one = sum(hasattr(item, a) for a in cls.attrs) == count
        test_two = item.__class__.__name__ == 'Database'
        return test_one and test_two

    @classmethod
    def create(cls, item: t.Any, **kwargs) -> t.Any:
        from pymongo.database import Database

        from superduperdb import CFG
        from superduperdb.db.base.build import build_vector_database
        from superduperdb.db.base.db import DB
        from superduperdb.db.mongodb.artifacts import MongoArtifactStore
        from superduperdb.db.mongodb.data_backend import MongoDataBackend
        from superduperdb.db.mongodb.metadata import MongoMetaDataStore

        if kwargs:
            raise ValueError('MongoDb creator accepts no parameters')
        if not isinstance(item, Database):
            raise TypeError('Expected Database but got {type(item)}')

        return DB(
            databackend=MongoDataBackend(conn=item.client, name=item.name),
            metadata=MongoMetaDataStore(conn=item.client, name=item.name),
            artifact_store=MongoArtifactStore(
                conn=item.client, name=f'_filesystem:{item.name}'
            ),
            vector_database=build_vector_database(CFG.vector_search.type),
        )


class SklearnTyper(DuckTyper):
    attrs = '_predict', 'fit', 'score', 'transform'
    count = 2

    @classmethod
    def create(cls, item: t.Any, **kwargs) -> t.Any:
        from sklearn.base import BaseEstimator

        from superduperdb.ext.sklearn.model import Estimator

        if not isinstance(item, BaseEstimator):
            raise TypeError('Expected BaseEstimator but got {type(item)}')

        kwargs['identifier'] = auto_identify(item)
        return Estimator(object=item, **kwargs)


class TorchTyper(DuckTyper):
    attrs = 'forward', 'parameters', 'state_dict', '_load_from_state_dict'

    @classmethod
    def create(cls, item: t.Any, **kwargs) -> t.Any:
        from torch import jit, nn

        from superduperdb.ext.torch.model import TorchModel

        if isinstance(item, nn.Module) or isinstance(item, jit.ScriptModule):
            return TorchModel(identifier=auto_identify(item), object=item, **kwargs)

        raise TypeError('Expected a Module but got {type(item)}')


def auto_identify(instance: t.Any) -> str:
    return instance.__class__.__name__.lower()


_DUCK_TYPES = MongoDbTyper, SklearnTyper, TorchTyper
superduper = DuckTyper.run
