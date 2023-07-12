import typing as t

from superduperdb.core.component import Component
from superduperdb.core.model import Model
from superduperdb.datalayer.base.query import Select
import dataclasses as dc


@dc.dataclass
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

    variety: t.ClassVar[str] = 'watcher'

    key: str
    model: t.Union[str, Model]
    select: t.Union[Select] = None
    features: t.Optional[t.Dict] = None
    max_chunk_size: int = 5000
    active: bool = True
    version: t.Optional[int] = None
    identifier: t.Optional[str] = None
    db: dc.InitVar[t.Optional[t.Any]] = None

    def __post_init__(self, db):
        if isinstance(self.model, str) and db is not None:
            self.model = db.load('model', self.model)

        if self.identifier is None:
            self.identifier = f'{self.model.identifier}/{self.key}'

    def cleanup(self, database) -> None:
        self.select.model_cleanup(database, model=self.model.identifier, key=self.key)

    def schedule_jobs(
        self, database, verbose=False, dependencies=(), distributed=False
    ):
        if not self.active:
            return
        return self.model.predict(
            X=self.key,
            db=database,
            select=self.select,
            distributed=distributed,
            max_chunk_size=self.max_chunk_size,
            dependencies=dependencies,
        )
