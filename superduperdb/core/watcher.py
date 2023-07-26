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
    select: t.Optional[Select] = None
    max_chunk_size: int = 5000
    active: bool = True
    version: t.Optional[int] = None
    identifier: t.Optional[str] = None
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)

    @property
    def child_components(self):
        return [('model', 'model')]

    def _on_create(self, db):
        if isinstance(self.model, str):
            self.model = db.load('model', self.model)

    def __post_init__(self):
        if self.identifier is None and self.model is not None:
            if isinstance(self.model, str):
                self.identifier = f'{self.model}/{self.key}'
            else:
                self.identifier = f'{self.model.identifier}/{self.key}'
        self.features = {}
        if hasattr(self.select, 'features'):
            self.features = self.select.features

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
            **self.predict_kwargs,
        )
