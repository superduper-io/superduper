import dataclasses as dc
import math
import typing as t
from abc import abstractmethod
from collections import defaultdict
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, as_completed, wait
from traceback import format_exc

import numpy
import pandas

from superduper import logging
from superduper.base import Base
from superduper.base.metadata import STATUS_FAILED
from superduper.base.schema import Schema
from superduper.misc.importing import import_object

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class Event(Base):
    """Base class for all events."""

    max_batch_size: t.ClassVar[int | None] = None

    @classmethod
    def batch_execute(
        cls, events: t.List['Event'], db: 'Datalayer', batch_size: int | None = None
    ):
        """Execute the events in batch.

        :param events: list of events.
        :param db: Datalayer instance.
        :param batch_size: size of the batch.
        """
        for event in events:
            event.execute(db)

    @abstractmethod
    def execute(
        self,
        db: 'Datalayer',
    ):
        """Execute the event.

        :param db: Datalayer instance.
        """
        raise NotImplementedError('Not relevant for this event class')


class Signal(Event):
    """
    Event used to send a signal to the scheduler.

    :param msg: signal to send
    :param context: the context of component creation
    """

    queue: t.ClassVar[str] = '_apply'

    msg: str
    context: str

    def execute(
        self,
        db: 'Datalayer',
    ):
        """Execute the signal.

        :param db: Datalayer instance.
        """
        if self.msg.lower() == 'done':
            db.cluster.compute.release_futures(self.context)


class Change(Event):
    """
    Class for streaming change events.

    :param ids: list of ids detected in databackend.
    :param type: {'insert', 'update', 'delete'}
    :param queue: which table was affected
    :param ids: the ids affected
    """

    type: str
    queue: str
    ids: t.List[str]

    def execute(
        self,
        db: 'Datalayer',
    ):
        """Execute the change event.

        :param db: Datalayer instance.
        """
        raise NotImplementedError('Not relevant for this event class')


class CreateTable(Event):
    """
    Class for table creation events.

    :param identifier: the identifier of the table
    :param primary_id: the primary id of the table
    :param fields: the schema of the table
    :param is_component: whether the table is a component
    """

    queue: t.ClassVar[str] = '_apply'
    max_batch_size: t.ClassVar[int] = 100

    identifier: str
    primary_id: str
    fields: t.Dict
    is_component: bool = False

    @property
    def huuid(self):
        """Return the hashed uuid."""
        schema_str = ';'.join([f'{k}={v}' for k, v in sorted(self.fields.items())])
        return f'{self.identifier}[{schema_str}]'

    @classmethod
    def batch_execute(
        cls,
        events: t.List['CreateTable'],
        db: 'Datalayer',
        batch_size: int | None = None,
    ):
        """Execute the create table events in batch.

        :param events: list of create table events.
        :param db: Datalayer instance.
        """
        batch_size = batch_size or cls.max_batch_size
        n_batches = len(events) // batch_size + 1
        for i in range(0, len(events), batch_size):
            iteration = i // batch_size
            logging.info(
                'Creating tables and schemas (batch {}/{})'.format(
                    iteration + 1, n_batches
                )
            )
            batch = events[i : i + cls.max_batch_size]
            db.metadata.create_tables_and_schemas(batch)
        logging.info('Created tables and schemas... DONE')

    def execute(self, db: 'Datalayer'):
        """Execute the create event.

        :param db: Datalayer instance.
        """
        return db.metadata.create_table_and_schema(
            identifier=self.identifier,
            primary_id=self.primary_id,
            schema=Schema.build(**self.fields),
            is_component=self.is_component,
        )


class Create(Event):
    """
    Class for component creation events.

    :param context: the component context of creation.
    :param path: path of the component to be created
    :param data: the data of the component
    :param parent: the parent of the component (if any)
    :param children: the children of the component (if any)
    """

    queue: t.ClassVar[str] = '_apply'
    max_batch_size: t.ClassVar[int] = 1000

    context: str
    path: str
    data: t.Dict
    parent: list | None = None
    children: t.List | None = None

    @property
    def component(self):
        return self.path.split('.')[-1]

    @staticmethod
    def cluster_by_component(events: t.List['Create']):
        """Cluster events by component.

        :param events: list of create events.
        :return: list of create events clustered by component.
        """
        clustered_events: t.Dict = defaultdict(list)
        for event in events:
            if event.component not in clustered_events:
                clustered_events[event.component] = []
            clustered_events[event.component].append(event)
        return clustered_events.values()

    def execute(self, db: 'Datalayer'):
        """Execute the create event.

        :param db: Datalayer instance.
        """
        logging.info(
            f'Creating {self.path.split("/")[-1]}:'
            f'{self.data["identifier"]}:{self.data["uuid"]}'
        )

        db.metadata.create_component(self.data, path=self.path)

        try:
            artifact_ids, _ = db._find_artifacts(self.data)

            db.metadata.create_artifact_relation(
                component=self.component,
                identifier=self.data['identifier'],
                uuid=self.data['uuid'],
                artifact_ids=artifact_ids,
            )

            if self.children:
                for child in self.children:
                    db.metadata.create_parent_child(
                        child_component=child[0],
                        child_identifier=child[1],
                        child_uuid=child[2],
                        parent_component=self.component,
                        parent_identifier=self.data['identifier'],
                        parent_uuid=self.data['uuid'],
                    )

            logging.info(
                f'Creating {self.path.split("/")[-1]}:'
                f'{self.data["identifier"]}:{self.data["uuid"]}... DONE'
            )

        except Exception as e:
            db.metadata.set_component_status(
                component=self.component,
                uuid=self.data['uuid'],
                details_update={
                    'phase': STATUS_FAILED,
                    'reason': f'Failed to create: {str(e)}',
                    'message': format_exc(),
                },
            )
            raise e

    @property
    def huuid(self):
        """Return the hashed uuid."""
        return f'{self.component}:' f'{self.data["identifier"]}:' f'{self.data["uuid"]}'


class PutComponent(Event):
    """
    Class for putting component on cluster.

    :param context: the component context of creation.
    :param component: the type of component to be created
    :param identifier: the identifier of the component to be created
    :param uuid: the uuid of the component to be created
    :param service: the service to put the component on
    """

    queue: t.ClassVar[str] = '_apply'
    timeout: t.ClassVar[int] = 30

    context: str
    component: str
    identifier: str
    uuid: str
    service: str

    @property
    def cls(self):
        """Return the class of the component."""
        return import_object(self.path)

    @property
    def huuid(self):
        return f'{self.component}:{self.identifier}:{self.uuid}/{self.service}'

    @classmethod
    def batch_execute(
        cls,
        events: t.List['PutComponent'],
        db: 'Datalayer',
        batch_size: int | None = None,
    ):
        """Execute the put component events in batch.

        :param events: list of put component events
        :param db: Datalayer instance.
        :param batch_size: size of the batch.
        """
        batch_size = batch_size or cls.max_batch_size
        if batch_size is None or batch_size == 1 or len(events) <= 1:
            return super().batch_execute(events, db, batch_size)  # type: ignore[arg-type]

        # helper so we can capture huuid in log messages
        def _run(event: "PutComponent"):
            event.execute(db)  # Any exception propagates via the Future
            logging.debug(f"put {event.huuid}")

        # How many chunks do we need?
        n_chunks = math.ceil(len(events) / batch_size)

        for i in range(0, len(events), batch_size):
            chunk = events[i : i + batch_size]
            logging.info(
                f"Putting {len(chunk)} components "
                f"(chunk {i // batch_size + 1}/{n_chunks})"
            )

            with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
                futs = {pool.submit(_run, ev): ev for ev in chunk}

                done, pending = wait(
                    futs,
                    timeout=cls.timeout,
                    return_when=FIRST_EXCEPTION,  # stop as soon as we see an error
                )

                first_exc = None
                for f in done:
                    try:
                        f.result()  # will re-raise if _run() failed
                    except Exception as exc:
                        if first_exc is None:
                            first_exc = exc

                if pending:
                    for f in pending:
                        f.cancel()  # polite request; running tasks may ignore
                    raise TimeoutError(
                        f"{len(pending)} of {len(chunk)} futures did not finish "
                        f"within {cls.timeout} s."
                    )

                # If something else failed, raise it *after* every task ended
                if first_exc is not None:
                    raise first_exc

    def execute(self, db: 'Datalayer'):
        """Execute the put on cluster event.

        :param db: Datalayer instance.
        """
        logging.info(
            f'Putting {self.component}:'
            f'{self.identifier}:{self.uuid} on {self.service}'
        )
        getattr(db.cluster, self.service).put_component(
            component=self.component,
            uuid=self.uuid,
        )
        logging.info(
            f'Putting {self.component}:'
            f'{self.identifier}:{self.uuid} on {self.service}'
            '... DONE'
        )


class Delete(Event):
    """
    Class for component deletion events.

    :param component: the type of component to be created
    :param identifier: the identifier of the component to be deleted
    :param parents: the parents of the component (if any)
    """

    queue: t.ClassVar[str] = '_apply'

    component: str
    identifier: str
    parents: t.List[str] = dc.field(default_factory=list)

    @property
    def huuid(self):
        """Return the hashed uuid."""
        return f'{self.component}:{self.identifier}'

    def execute(self, db: 'Datalayer'):
        """Execute the delete event.

        :param db: Datalayer instance.
        """
        try:
            object = db.load(component=self.component, identifier=self.identifier)
            object.cleanup()
            db.metadata.delete_component(self.component, self.identifier)
            artifact_ids = db.metadata.get_artifact_relations_for_component(
                self.component, self.identifier
            )

            if artifact_ids:
                parents_to_artifacts = db.metadata.get_artifact_relations_for_artifacts(
                    artifact_ids
                )
                df = pandas.DataFrame(parents_to_artifacts)
                if not df.empty:
                    condition = numpy.logical_or(
                        df['component'] != self.component,
                        df['identifier'] != self.identifier,
                    )
                    other_relations = df[condition]
                    to_exclude = other_relations['artifact_id'].tolist()
                    artifact_ids = sorted(list(set(artifact_ids) - set(to_exclude)))
                db.artifact_store.delete_artifact(artifact_ids)

            db.metadata.delete_parent_child_relationships(
                parent_component=self.component,
                parent_identifier=self.identifier,
            )

        except Exception as e:
            try:
                db.metadata.set_component_failed(
                    component=self.component,
                    uuid=object.uuid,
                    reason=f'Failed to delete: {str(e)}',
                    message=str(format_exc()),
                    context=None,
                )
            except Exception as ee:
                logging.error(
                    f'Failed to set component status: {str(ee)}'
                    f'while deleting {self.component}:{self.identifier}'
                )
            raise e


class Update(Event):
    """
    Update component event.

    :param context: the component context of creation.
    :param component: the type of component to be created
    :param data: the component data to be created
    :param parent: the parent of the component (if any)
    """

    queue: t.ClassVar[str] = '_apply'

    context: str
    component: str
    data: t.Dict
    parent: list | None = None

    def execute(
        self,
        db: 'Datalayer',
    ):
        """Execute the create event.

        :param db: Datalayer instance.
        """
        try:
            artifact_ids, _ = db._find_artifacts(self.data)
            db.metadata.create_artifact_relation(
                component=self.component,
                identifier=self.data['identifier'],
                uuid=self.data['uuid'],
                artifact_ids=artifact_ids,
            )
            db.metadata.replace_object(
                self.component, uuid=self.data['uuid'], info=self.data
            )
        except Exception as e:
            db.metadata.set_component_status(
                component=self.component,
                uuid=self.data['uuid'],
                details_update={
                    'phase': STATUS_FAILED,
                    'reason': f'Failed to update: {str(e)}',
                    'message': format_exc(),
                },
            )
            raise e

    @property
    def huuid(self):
        """Return the hashed uuid."""
        return f'{self.component}' f'{self.data["identifier"]}:' f'{self.data["uuid"]}'


def unpack_event(r):
    """
    Helper function to deserialize event into Event class.

    :param r: Serialized event.
    """
    return Base.decode(r)
