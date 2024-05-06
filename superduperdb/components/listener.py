import dataclasses as dc
import typing as t

from overrides import override

from superduperdb import CFG
from superduperdb.backends.base.query import Query
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import _OUTPUTS_KEY
from superduperdb.components.model import Mapping
from superduperdb.misc.annotations import public_api
from superduperdb.misc.server import request_server
from superduperdb.rest.utils import parse_query

from ..jobs.job import Job
from .component import Component, ComponentTuple
from .model import Model, ModelInputType

SELECT_TEMPLATE = {'documents': [], 'query': '<collection_name>.find()'}


@public_api(stability='stable')
@dc.dataclass(kw_only=True)
class Listener(Component):
    """
    Listener object which is used to process a column/ key of a collection or table,
    and store the outputs.
    {component_parameters}
    :param key: Key to be bound to model
    :param model: Model for processing data
    :param select: Object for selecting which data is processed
    :param identifier: A string used to identify the model.
    :param active: Toggle to ``False`` to deactivate change data triggering
    :param predict_kwargs: Keyword arguments to self.model.predict
    """

    __doc__ = __doc__.format(component_parameters=Component.__doc__)

    ui_schema: t.ClassVar[t.List[t.Dict]] = [
        {'name': 'identifier', 'type': 'str', 'default': ''},
        {'name': 'key', 'type': 'json'},
        {'name': 'model', 'type': 'component/model'},
        {'name': 'select', 'type': 'json', 'default': SELECT_TEMPLATE},
        {'name': 'active', 'type': 'bool', 'default': True},
        {'name': 'predict_kwargs', 'type': 'json', 'default': {}},
    ]

    key: ModelInputType
    model: Model
    select: Query 
    active: bool = True
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)
    identifier: str = ''

    type_id: t.ClassVar[str] = 'listener'

    @classmethod
    def handle_integration(cls, kwargs):
        if 'select' in kwargs and isinstance(kwargs['select'], dict):
            kwargs['select'] = parse_query(
                query=kwargs['select']['query'],
                documents=kwargs['select']['documents'],
            )
        return kwargs

    def __post_init__(self, db, artifacts):
        if self.identifier == '':
            self.identifier = self.id
        super().__post_init__(db, artifacts)

    @property
    def id(self):
        return f'component/{self.type_id}/{self.model.identifier}/{self.uuid}'

    @property
    def mapping(self):
        return Mapping(self.key, signature=self.model.signature)

    @property
    def outputs(self):
        return f'{_OUTPUTS_KEY}.{self.uuid}'

    @property
    def outputs_select(self):
        return self.select.table_or_collection.outputs(self.id)

    @property
    def outputs_key(self):
        if self.select.DB_TYPE == "SQL":
            return 'output'
        else:
            return self.outputs

    @override
    def post_create(self, db: Datalayer) -> None:
        output_table = db.databackend.create_output_dest(
            self.uuid,
            self.model.datatype,
            flatten=self.model.flatten,
        )
        if output_table is not None:
            db.add(output_table)
        if self.select is not None and self.active and not db.server_mode:
            if CFG.cluster.cdc.uri:
                request_server(
                    service='cdc',
                    endpoint='listener/add',
                    args={'name': self.identifier},
                    type='get',
                )
            else:
                db.cdc.add(self)

    @property
    def dependencies(self) -> t.List[ComponentTuple]:
        args, kwargs = self.mapping.mapping
        all_ = list(args) + list(kwargs.values())
        out = []
        for x in all_:
            if x.startswith('_outputs.'):
                listener_id = x.split('.')[1]
                out.append(listener_id)
        return out

    def depends_on(self, other: Component):
        if not isinstance(other, Listener):
            return False

        args, kwargs = self.mapping.mapping
        all_ = list(args) + list(kwargs.values())

        return any([x.startswith(f'_outputs.{other.uuid}') for x in all_])

    @override
    def schedule_jobs(
        self, db: Datalayer, dependencies: t.Sequence[Job] = (), overwrite: bool = False
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
            self.model.predict_in_db_job(
                X=self.key,
                db=db,
                predict_id=self.uuid,
                select=self.select,
                dependencies=dependencies,
                overwrite=overwrite,
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
