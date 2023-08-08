import dataclasses as dc
import typing as t

from overrides import override

from superduperdb.db.base.db import DB
from superduperdb.db.base.query import Select

from .component import Component
from .job import Job
from .model import Model


@dc.dataclass
class Listener(Component):
    """
    Listener object which is used to process a column/ key of a collection or table,
    and store the outputs.

    :param key: Key to be bound to model
    :param model: Model for processing data

    :param active: Toggle to ``False`` to deactivate change data triggering
    :param features: Dictionary of mappings from keys to model
    :param identifier: A string used to identify the model.
    :param predict_kwargs: Keyword arguments to self.model.predict
    :param select: Object for selecting which data is processed
    :param version: Version number of the model(?)
    """

    key: str
    model: t.Union[str, Model]

    active: bool = True
    features: t.Optional[t.Dict] = None
    identifier: t.Optional[str] = None
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)
    select: t.Optional[Select] = None
    version: t.Optional[int] = None

    #: A unique name for the class
    type_id: t.ClassVar[str] = 'listener'

    @property
    def child_components(self) -> t.Sequence[t.Tuple[str, str]]:
        """Returns a list of child components as pairs TBD"""
        return [('model', 'model')]

    @override
    def on_create(self, db: DB) -> None:
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
            self.features = self.select.features  # type: ignore[union-attr]

    @override
    def schedule_jobs(
        self,
        database: DB,
        dependencies: t.Sequence[Job] = (),
        distributed: bool = False,
        verbose: bool = False,
    ) -> t.Sequence[t.Any]:
        if not self.active:
            return ()

        return self.model.predict(  # type: ignore[union-attr]
            X=self.key,
            db=database,
            select=self.select,
            distributed=distributed,
            dependencies=dependencies,
            **(self.predict_kwargs or {}),
        )

    def cleanup(self, database: DB) -> None:
        """Clean up when the listener is done

        :param database: The db to process
        """

        self.select.model_cleanup(  # type: ignore[union-attr]
            database,
            model=self.model.identifier,  # type: ignore[union-attr]
            key=self.key,
        )
