import dataclasses as dc
import typing as t

from overrides import override

from superduper import CFG, logging
from superduper.backends.base.query import Query
from superduper.base.document import _OUTPUTS_KEY
from superduper.components.model import Mapping
from superduper.misc.server import request_server

from ..jobs.job import Job
from .component import Component
from .model import Model, ModelInputType

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


SELECT_TEMPLATE = {'documents': [], 'query': '<collection_name>.find()'}


class Listener(Component):
    """Listener component.

    Listener object which is used to process a column/key of a collection or table,
    and store the outputs.

    :param key: Key to be bound to the model.
    :param model: Model for processing data.
    :param select: Object for selecting which data is processed.
    :param predict_kwargs: Keyword arguments to self.model.predict().
    :param identifier: A string used to identify the model.
    """

    key: ModelInputType
    model: Model
    select: t.Union[Query, None]
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)
    identifier: str = ''

    type_id: t.ClassVar[str] = 'listener'

    def __post_init__(self, db, artifacts):
        if self.identifier == '':
            self.identifier = self.uuid
        super().__post_init__(db, artifacts)

    @property
    def mapping(self):
        """Mapping property."""
        return Mapping(self.key, signature=self.model.signature)

    @property
    def outputs(self):
        """Get reference to outputs of listener model."""
        return f'{_OUTPUTS_KEY}.{self.uuid}'

    @property
    def outputs_key(self):
        """Model outputs key."""
        logging.warn(
            (
                "listener.outputs_key is deprecated and will be removed"
                "in a future release. Please use listener.outputs instead."
            )
        )
        return self.outputs

    @property
    def outputs_select(self):
        """Get select statement for outputs."""
        return self.db[self.select.table].select().outputs(self.predict_id)

    @override
    def post_create(self, db: "Datalayer") -> None:
        """Post-create hook.

        :param db: Data layer instance.
        """
        self.create_output_dest(db, self.uuid, self.model)
        if self.select is not None:  # and not db.server_mode:
            logging.info('Requesting listener setup on CDC service')
            if CFG.cluster.cdc.uri:
                logging.info('Sending request to add listener')
                request_server(
                    service='cdc',
                    endpoint='listener/add',
                    args={'name': self.identifier},
                    type='get',
                )
            else:
                logging.info(
                    'Skipping listener setup on CDC service since no URI is set'
                )
        else:
            logging.info(
                'Skipping listener setup on CDC service'
                f' since select is {self.select} or server mode is {db.server_mode}'
            )
        db.compute.queue.declare_component(self)

    @classmethod
    def create_output_dest(cls, db: "Datalayer", uuid, model: Model):
        """
        Create output destination.

        :param db: Data layer instance.
        :param uuid: UUID of the listener.
        :param model: Model instance.
        """
        if model.datatype is None and model.output_schema is None:
            return
        output_table = db.databackend.create_output_dest(
            uuid,
            model.datatype,
            flatten=model.flatten,
        )
        if output_table is not None:
            db.add(output_table)

    @property
    def dependencies(self):
        """Listener model dependencies."""
        args, kwargs = self.mapping.mapping
        all_ = list(args) + list(kwargs.values())
        out = []
        for x in all_:
            if x.startswith('_outputs.'):
                listener_id = x.split('.')[1]
                out.append(listener_id)
        return out

    @property
    def predict_id(self):
        """Get predict ID."""
        return self.uuid

    def trigger_ids(self, query: Query, primary_ids: t.Sequence):
        """Get trigger IDs.

        Only the ids returned by this function will trigger the listener.

        :param query: Query object.
        :param primary_ids: Primary IDs.
        """
        conditions = [
            # trigger by main table
            self.select and self.select.table == query.table,
            # trigger by output table
            query.table in self.key and query.table != self.outputs,
        ]
        if not any(conditions):
            return []

        if self.select is None:
            return []

        if self.select.table == query.table:
            trigger_ids = list(primary_ids)

        else:
            trigger_ids = [
                doc['_source'] for doc in query.documents if '_source' in doc
            ]

        # TODO: Use the better way to detect the keys exist in the trigger_ids
        data = self.db.execute(self.select.select_using_ids(trigger_ids))
        keys = self.key

        if isinstance(self.key, str):
            keys = [self.key]
        elif isinstance(self.key, dict):
            keys = list(self.key.keys())

        ready_ids = []
        for select in data:
            notfound = 0
            for k in keys:
                try:
                    select[k]
                except KeyError:
                    notfound += 1
            if notfound == 0:
                ready_ids.append(select[self.select.primary_id])
        return ready_ids

    @override
    def schedule_jobs(
        self,
        db: "Datalayer",
        dependencies: t.Sequence[Job] = (),
        overwrite: bool = False,
        ids: t.Optional[t.List[t.Any]] = None,
    ) -> t.Sequence[t.Any]:
        """Schedule jobs for the listener.

        :param db: Data layer instance to process.
        :param dependencies: A list of dependencies.
        :param overwrite: Overwrite the existing data.
        :param ids: Optional ids to schedule.
        """
        if self.select is None:
            return []
        from superduper.base.datalayer import DBEvent
        from superduper.base.event import Event

        events = []
        if ids is None:
            ids = db.execute(self.select.select_ids)
            ids = [id[self.select.primary_id] for id in ids]

        events = [
            Event(
                type_id=self.type_id,
                identifier=self.identifier,
                event_type=DBEvent.insert,
                id=str(id),
            )
            for id in ids
        ]

        return db.compute.broadcast(events)

    def listener_dependencies(self, db, deps: t.Sequence[str]) -> t.Sequence[str]:
        """List all dependent job ids of the listener."""
        dependencies_ids: t.Sequence[str] = []
        for predict_id in self.dependencies:
            try:
                upstream_listener = db.load(uuid=predict_id)
                upstream_model = upstream_listener.model
            except Exception:
                logging.warn(
                    f"Could not find the upstream listener with uuid {predict_id}"
                )
                continue
            job_ids = upstream_model.jobs(db=db)
            dependencies_ids.extend(job_ids)

        dependencies = tuple([*dependencies_ids, *deps])
        return dependencies

    def run_jobs(
        self,
        db: "Datalayer",
        dependencies: t.Sequence[str] = (),
        overwrite: bool = False,
        ids: t.Optional[t.List] = [],
        event_type: str = 'insert',
    ) -> t.Sequence[t.Any]:
        """Schedule jobs for the listener.

        :param db: Data layer instance to process.
        :param dependencies: A list of dependencies.
        :param overwrite: Overwrite the existing data.
        :param event_type: Type of event.
        """
        if self.select is None:
            return []
        assert not isinstance(self.model, str)

        dependencies = self.listener_dependencies(db, dependencies)

        out = [
            self.model.predict_in_db_job(
                X=self.key,
                db=db,
                predict_id=self.uuid,
                select=self.select,
                ids=ids,
                dependencies=dependencies,
                overwrite=overwrite,
                **(self.predict_kwargs or {}),
            )
        ]
        return out

    def cleanup(self, db: "Datalayer") -> None:
        """Clean up when the listener is deleted.

        :param db: Data layer instance to process.
        """
        if self.select is not None:
            self.db[self.select.table].drop_outputs(self.predict_id)
