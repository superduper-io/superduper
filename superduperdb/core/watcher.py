import typing as t

from superduperdb.core.base import Component, Placeholder
from superduperdb.core.model import Model
from superduperdb.datalayer.base.query import Select
from superduperdb.misc.serialization import from_dict, to_dict


class Watcher(Component):
    """
    Watcher object which is used to process a column/ key of a collection or table,
    and store the outputs.

    :param select: Object for selecting which data is processed
    :param model: Model for processing data
    :param key: Key to be bound to model
    :param features: Dictionary of mappings from keys to models
    :param active: Toggle to ``False`` to deactivate change data triggering
    """

    features: t.Dict
    key: str
    model: t.Union[Placeholder, Model]
    select: Select
    variety = 'watcher'
    max_chunk_size: int

    def __init__(
        self,
        select: Select,
        model: t.Union[Model, str],
        key: str = '_base',
        features: t.Optional[t.Dict] = None,
        active: bool = True,
        max_chunk_size: int = 5000,
    ):
        self.model = model if isinstance(model, Model) else Placeholder(model, 'model')
        self.select = select
        self.key = key
        self.features = features or {}
        self.active = active
        self.max_chunk_size = max_chunk_size
        identifier = f'{self.model.identifier}/{self.key}'
        super().__init__(identifier)

    def asdict(self) -> t.Dict[str, t.Any]:
        return {
            'model': self.model.identifier,
            'select': to_dict(self.select),
            'key': self.key,
            'identifier': self.identifier,
            'features': self.features or {},
            'active': self.active,
        }

    @staticmethod
    def cleanup(info, database) -> None:
        select = from_dict(info['select'])
        select.model_cleanup(database, model=info['model'], key=info['key'])

    def schedule_jobs(self, database, verbose=False, dependencies=(), remote=False):
        if not self.active:
            return
        return self.model.predict(
            X=self.key,
            db=database,
            select=self.select,
            remote=remote,
            max_chunk_size=self.max_chunk_size,
            dependencies=dependencies,
        )
