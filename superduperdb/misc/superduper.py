import typing as t

__all__ = ('superduper',)


class DuckTyper:
    attrs: t.Sequence[str]
    count: int = 0
    DUCK_TYPES: t.Tuple[str] = ()

    @classmethod
    def run(cls, item: t.Any, **kwargs):
        dts = [dt for dt in cls.DUCK_TYPES if dt.accept(item)]
        if len(dts) == 1:
            return cls.create(item, **kwargs)
        raise NotImplementedError(
            f'Couldn\'t auto-identify {item}, please wrap explicitly using '
            '``superduperdb.core.*``'
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
    def create(cls, item: t.Any, **kwargs) -> t.Any:
        from pymongo.database import Database
        from superduperdb import CFG
        from superduperdb.datalayer.base.build import build_vector_database
        from superduperdb.datalayer.base.datalayer import Datalayer
        from superduperdb.datalayer.mongodb.artifacts import MongoArtifactStore
        from superduperdb.datalayer.mongodb.data_backend import MongoDataBackend
        from superduperdb.datalayer.mongodb.metadata import MongoMetaDataStore

        if kwargs:
            raise ValueError('MongoDb creator accepts no parameters')
        if not isinstance(item, Database):
            raise TypeError('Expected Database but got {type(item)}')

        return Datalayer(
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
        from sklearn.pipeline import Pipeline as BasePipeline
        from superduperdb.models.sklearn.wrapper import Estimator, Pipeline

        if not isinstance(item, BaseEstimator):
            raise TypeError('Expected BaseEstimator but got {type(item)}')

        kwargs['identifier'] = auto_identify(item)
        if isinstance(item, BasePipeline):
            return Pipeline(steps=item.steps, memory=item.memory, **kwargs)
        else:
            return Estimator(estimator=item, **kwargs)


class TorchTyper(DuckTyper):
    attrs = 'forward', 'parameters', 'state_dict'

    @classmethod
    def create(cls, item: t.Any, **kwargs) -> t.Any:
        from superduperdb.models.torch.wrapper import TorchModel
        from torch import nn, jit

        if isinstance(item, nn.Module) or isinstance(item, jit.ScriptModule):
            return TorchModel(identifier=auto_identify(item), object=item, **kwargs)

        raise TypeError('Expected a Module but got {type(item)}')


def auto_identify(instance):
    return instance.__class__.__name__.lower()


DuckTyper.DUCK_TYPES = MongoDbTyper, SklearnTyper, TorchTyper
superduper = DuckTyper.run
