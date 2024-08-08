import dataclasses as dc
import typing as t

from overrides import override

from superduper import CFG, logging
from superduper.base.event import Event
from superduper.backends.base.query import Query
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
    :param predict_id: A string used to identify the model's output.
    """

    key: ModelInputType
    model: Model
    select: t.Union[Query, None]
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)
    identifier: str = ''
    predict_id: str = ''

    type_id: t.ClassVar[str] = 'listener'

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)
        if not self.predict_id:
            self.predict_id = (
                self.identifier
                if self.version is None
                else f"{self.identifier}::{self.version}"
            )

    @property
    def mapping(self):
        """Mapping property."""
        return Mapping(self.key, signature=self.model.signature)

    @property
    def outputs(self):
        """Get reference to outputs of listener model."""
        return f'{CFG.output_prefix}{self.predict_id}'

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

    @property
    def cdc_table(self):
        """Get table for cdc."""
        return self.select.table_or_collection.identifier

    @override
    def post_create(self, db: "Datalayer") -> None:
        """Post-create hook.

        :param db: Data layer instance.
        """
        self.create_output_dest(db, self.predict_id, self.model)
        if self.select is not None:
            logging.info('Requesting listener setup on CDC service')
            if CFG.cluster.cdc.uri:
                logging.info('Sending request to add listener')
                request_server(
                    service='cdc',
                    endpoint='component/add',
                    args={'name': self.identifier, 'type_id': self.type_id},
                    type='get',
                )
            else:
                logging.info(
                    'Skipping listener setup on CDC service since no URI is set'
                )
        else:
            logging.info('Skipping listener setup on CDC service')
        db.compute.queue.declare_component(self)

    @classmethod
    def create_output_dest(cls, db: "Datalayer", predict_id, model: Model):
        """
        Create output destination.

        :param db: Data layer instance.
        :param uuid: UUID of the listener.
        :param model: Model instance.
        """
        if model.datatype is None and model.output_schema is None:
            return
        output_table = db.databackend.create_output_dest(
            predict_id,
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
            if x.startswith(CFG.output_prefix):
                listener_id = x[len(CFG.output_prefix) :].split(".")[0]
                out.append(listener_id)
        return out

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

        return self.db.databackend.check_ready_ids(
            self.select, self._ready_keys, trigger_ids
        )

    @property
    def _ready_keys(self):
        keys = self.key

        if isinstance(self.key, str):
            keys = [self.key]
        elif isinstance(self.key, dict):
            keys = list(self.key.keys())

        # Support multiple levels of nesting
        clean_keys = []
        for key in keys:
            if key.startswith(CFG.output_prefix):
                key = CFG.output_prefix + key[len(CFG.output_prefix) :].split(".")[0]
            else:
                key = key.split(".")[0]

            clean_keys.append(key)

        return clean_keys

    @override
    def schedule_jobs(
        self,
        db: "Datalayer",
        dependencies: t.Sequence[Job] = (),
        overwrite: bool = False,
    ) -> t.Sequence[t.Any]:
        """Schedule jobs for the listener.

        :param db: Data layer instance to process.
        :param dependencies: A list of dependencies.
        :param overwrite: Overwrite the existing data.
        """
        if self.select is None:
            return []
        from superduper.base.datalayer import DBEvent
        from superduper.base.event import Event

        events = []

        self.select.db = db
        ids = self.db.databackend.check_ready_ids(self.select, self._ready_keys)

        event = Event(
            dest={'type_id': self.type_id, 'identifier': self.identifier},
            event_type=DBEvent.insert,
            id=ids,
            on='COMPONENT',
        )

        db.compute.broadcast([event])
        return [event.uuid]

    def listener_dependencies(self, db, deps: t.Sequence[str]) -> t.Sequence[str]:
        """List all dependent job ids of the listener."""
        dependencies_ids: t.Sequence[str] = []
        predict_id2uuid = {}
        for listener_identifier in db.show('listener'):
            listener_info = db.metadata.get_component(
                type_id='listener',
                identifier=listener_identifier,
            )
            predict_id2uuid[listener_info['predict_id']] = listener_info['uuid']

        for predict_id in self.dependencies:
            if predict_id not in predict_id2uuid:
                logging.warn(
                    f"Could not find the upstream listener with predict_id {predict_id}"
                )
                continue
            uuid = predict_id2uuid.get(predict_id)
            upstream_listener = db.load(uuid=uuid)
            upstream_model = upstream_listener.model
            job_ids = upstream_model.jobs(db=db)
            dependencies_ids.extend(job_ids)

        dependencies = tuple([*dependencies_ids, *deps])
        return dependencies

    def run_jobs(
        self,
        db: "Datalayer",
        dependencies: t.Sequence[str] = (),
        overwrite: bool = False,
        events: t.Optional[t.List] = [],
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
        component_events, db_events = Event.chunk_by_type(events)

        # Create a startup job
        jobs = []
        for event in component_events:
            jobs += [
                self._create_predict_job(
                    db=db,
                    ids=event.id,
                    deps=dependencies,
                    overwrite=overwrite,
                    job_id=event.uuid,
                )
            ]

        jobs += [
            self._create_predict_job(
                db=db,
                ids=[event.id for event in db_events],
                deps=dependencies,
                overwrite=overwrite,
            )
        ]
        return jobs

    def _create_predict_job(self, db, ids, deps, overwrite, job_id=None):
        return self.model.predict_in_db_job(
            X=self.key,
            db=db,
            predict_id=self.uuid,
            select=self.select,
            ids=ids,
            job_id=job_id,
            dependencies=deps,
            overwrite=overwrite,
            **(self.predict_kwargs or {}),
        )

    def cleanup(self, db: "Datalayer") -> None:
        """Clean up when the listener is deleted.

        :param db: Data layer instance to process.
        """
        if self.select is not None:
            self.db[self.select.table].drop_outputs(self.predict_id)
