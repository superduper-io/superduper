import dataclasses as dc
import typing as t

from overrides import override

from superduperdb import CFG
from superduperdb.backends.base.query import CompoundSelect
from superduperdb.base.datalayer import Datalayer
from superduperdb.misc.server import request_server

from ..jobs.job import Job
from .component import Component
from .model import Model


@dc.dataclass
class Listener(Component):
    """
    Listener object which is used to process a column/ key of a collection or table,
    and store the outputs.

    :param key: Key to be bound to model
    :param model: Model for processing data
    :param select: Object for selecting which data is processed
    :param active: Toggle to ``False`` to deactivate change data triggering
    :param identifier: A string used to identify the model.
    :param predict_kwargs: Keyword arguments to self.model.predict
    :param version: Version number of the model(?)
    """

    key: str
    model: t.Union[str, Model]
    select: CompoundSelect
    active: bool = True
    identifier: t.Optional[str] = None
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)

    # Don't set this manually
    version: t.Optional[int] = None

    type_id: t.ClassVar[str] = 'listener'

    @property
    def child_components(self) -> t.Sequence[t.Tuple[str, str]]:
        """Returns a list of child components as pairs TBD"""
        return [('model', 'model')]

    @override
    def pre_create(self, db: Datalayer) -> None:
        if isinstance(self.model, str):
            self.model = t.cast(Model, db.load('model', self.model))
        if self.select is not None and self.select.variables:
            self.select = t.cast(CompoundSelect, self.select.set_variables(db))

    def post_create(self, db: Datalayer) -> None:
        # Start cdc service if enabled
        if self.select is not None and self.active and not db.server_mode:
            if CFG.cluster.cdc:
                request_server(
                    service='cdc',
                    endpoint='listener/add',
                    args={'name': self.identifier},
                    type='get',
                )
            else:
                db.cdc.add(self)

    @property
    def dependencies(self) -> t.List[str]:
        out = []
        if self.key.startswith('_outputs.'):
            _, key, model = self.key.split('.')
            out.append(f'{model}/{key}')
        if self.select.output_fields:
            out.extend([f'{v}/{k}' for k, v in self.select.output_fields.items()])
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

    @override
    def schedule_jobs(
        self,
        database: Datalayer,
        dependencies: t.Sequence[Job] = (),
        verbose: bool = False,
    ) -> t.Sequence[t.Any]:
        """
        Schedule jobs for the listener

        :param database: The DB instance to process
        :param dependencies: A list of dependencies
        :param verbose: Whether to print verbose output
        """
        if not self.active:
            return []
        assert not isinstance(self.model, str)

        out = [
            self.model.predict(
                X=self.key,
                db=database,
                select=self.select,
                dependencies=dependencies,
                **(self.predict_kwargs or {}),
            )
        ]
        return out

    def cleanup(self, database: Datalayer) -> None:
        """Clean up when the listener is deleted

        :param database: The DB instance to process
        """
        # TODO - this doesn't seem to do anything
        if (cleanup := getattr(self.select, 'model_cleanup', None)) is not None:
            assert not isinstance(self.model, str)
            cleanup(database, model=self.model.identifier, key=self.key)
