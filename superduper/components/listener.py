import dataclasses as dc
import typing as t

from overrides import override

from superduper import CFG, logging
from superduper.backends.base.query import Query
from superduper.base.datalayer import Datalayer
from superduper.base.event import Event
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
    :param identifier: A string used to identify the listener and it's outputs.
    """

    key: ModelInputType
    model: Model
    select: t.Union[Query, None]
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)
    type_id: t.ClassVar[str] = 'listener'

    def __post_init__(self, db, artifacts):
        deps = self.dependencies
        if deps:
            if not self.upstream:
                self.upstream = []
            for identifier, uuid in self.dependencies:
                self.upstream.append(f'&:component:listener:{identifier}:{uuid}')
        return super().__post_init__(db, artifacts)

    @property
    def predict_id(self):
        return f'{self.identifier}__{self.uuid}'

    def pre_create(self, db: Datalayer) -> None:
        return super().pre_create(db)

    @property
    def mapping(self):
        """Mapping property."""
        return Mapping(self.key, signature=self.model.signature)

    # TODO - do we need the outputs-prefix?
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
            if CFG.cluster.cdc.uri and not self.dependencies:
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
        if model.datatype is None:
            return
        # TODO make this universal over databackends
        # not special e.g. MongoDB vs. Ibis creating a table or not
        output_table = db.databackend.create_output_dest(
            predict_id,
            model.datatype,
            flatten=model.flatten,
        )

        db.apply(output_table)

    # TODO rename
    @property
    def dependencies(self):
        """Listener model dependencies."""
        args, kwargs = self.mapping.mapping
        all_ = list(args) + list(kwargs.values())
        out = []
        for x in all_:
            if x.startswith(CFG.output_prefix):
                out.append(tuple(x[len(CFG.output_prefix):].split('__')))
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

        self.select.db = db
        ids = self.db.databackend.check_ready_ids(self.select, self._ready_keys)
        if not ids:
            return []

        event = Event(
            dest={'type_id': self.type_id, 'identifier': self.identifier},
            event_type=DBEvent.insert,
            id=[str(id) for id in ids],
            from_type='COMPONENT',
            dependencies=dependencies,
            method='run'
        )

        db.compute.broadcast([event])
        return [event.uuid]

    def listener_dependencies(self, db: 'Datalayer', deps: t.Sequence[str]) -> t.Sequence[str]:
        """List all dependent job ids of the listener."""
        these_deps = self.dependencies
        if not these_deps:
            return deps

        deps = list(deps)

        for dep in these_deps:
            # TODO this logic is wrong
            # we should only get the jobs of the specific version
            identifier = db.show(type_id='listener', identifier=dep, uuid=dep.split('_')[-1])['identifier']
            jobs = db.metadata.show_jobs(type_id='listener', component_identifier=identifier)
            if not jobs:
                # Currently listener jobs are registered as belonging to the model
                # This is wrong and should be fixed
                # These dependencies won't work until this is fixed
                logging.warn(f'No jobs found for listener {dep}; something is wrong. Is this because we haven\'t refactored'
                             ' as listener jobs yet?')
            deps.extend(jobs)
        return list(set(deps))

    def run_jobs(
        self,
        db: "Datalayer",
        dependencies: t.Sequence[str] = (),
        overwrite: bool = False,
        events: t.Optional[t.List] = [],
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


        upstream_uuids = [up.split(':')[-1] for up in dependencies]
        upstream_jobs = db.metadata.show_job_ids(uuids=upstream_uuids, status='running')
        component_events, db_events = Event.chunk_by_type(events)

        # Create a startup job
        jobs = []
        for event in component_events:
            jobs += [
                self._create_predict_job(
                    db=db,
                    ids=event.id,
                    deps=[*upstream_jobs, *dependencies],
                    overwrite=overwrite,
                    job_id=event.uuid,
                )
            ]

        # Create db events
        if not db_events:
            return jobs

        jobs += [
            self._create_predict_job(
                db=db,
                ids=[event.id for event in db_events],
                deps=[*upstream_jobs, *dependencies],
                overwrite=overwrite,
            )
        ]
        return jobs

    def run(self, ids, db: t.Optional["Datalayer"] = None):
        return self.model.predict_in_db(
            X=self.key,
            db=self.db or db,
            predict_id=self.predict_id,
            select=self.select,
            ids=ids,
            **(self.predict_kwargs or {}),
        )

    def _create_predict_job(self, db, ids, deps, overwrite, job_id=None):
        return self.model.predict_in_db_job(
            X=self.key,
            db=db,
            predict_id=self.predict_id,
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
