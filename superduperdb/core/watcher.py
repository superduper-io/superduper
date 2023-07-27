import dataclasses as dc
import typing as t

from overrides import override

from superduperdb.datalayer.base.datalayer import Datalayer
from superduperdb.datalayer.base.query import Select

from .component import Component
from .job import Job
from .model import Model


@dc.dataclass
class Watcher(Component):
    """
    Watcher object which is used to process a column/ key of a collection or table,
    and store the outputs.

    :param key: Key to be bound to model
    :param model: Model for processing data

    :param active: Toggle to ``False`` to deactivate change data triggering
    :param features: Dictionary of mappings from keys to models
    :param identifier: A string used to identify the model.
    :param max_chunk_size: int = 5000

    :param predict_kwargs: Keyword arguments to self.model.predict
    :param select: Object for selecting which data is processed
    :param version: Version number of the model(?)
    """

    key: str
    model: t.Union[str, Model]

    active: bool = True
    features: t.Optional[t.Dict] = None
    identifier: t.Optional[str] = None
    max_chunk_size: int = 5000
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)
    select: t.Optional[Select] = None
    version: t.Optional[int] = None

    variety: t.ClassVar[str] = 'watcher'

    @property
    def child_components(self) -> t.Sequence[t.Tuple[str, str]]:
        """Returns a list of child components as pairs TBD"""
        return [('model', 'model')]

    @override
    def on_create(self, db: Datalayer) -> None:
        if isinstance(self.model, str):
            self.model = t.cast(Model, db.load('model', self.model))

    @property
    def dependencies(self) -> t.List[str]:
        out = []
        if self.features:
            for k in self.features:
                out.append(f'{self.features[k]}/{k}')
        if self.key.startswith('_outputs.'):
            _, key, model = self.key.split('.')
            out.append(f'{model}/{key}')
        return out

    @property
    def id_key(self) -> str:
        if self.key.startswith('_outputs.'):
            return self.key.split('.')[1]
        return self.key

    def __post_init__(self):
        if self.identifier is None and self.model is not None:
            if isinstance(self.model, str):
                self.identifier = f'{self.model}/{self.id_key}'
            else:
                self.identifier = f'{self.model.identifier}/{self.id_key}'
        self.features = {}
        if hasattr(self.select, 'features'):
            self.features = self.select.features

    @override
    def schedule_jobs(
        self,
        database: Datalayer,
        dependencies: t.Sequence[Job] = (),
        distributed: bool = False,
        verbose: bool = False,
    ) -> t.Sequence[t.Any]:
        if not self.active:
            return ()

        return self.model.predict(
            X=self.key,
            db=database,
            select=self.select,
            distributed=distributed,
            max_chunk_size=self.max_chunk_size,
            dependencies=dependencies,
            **(self.predict_kwargs or {}),
        )

    def cleanup(self, database: Datalayer) -> None:
        """Clean up when the watcher is done

        :param database: The datalayer to process
        """

        self.select.model_cleanup(database, model=self.model.identifier, key=self.key)
