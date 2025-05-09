import typing as t
from traceback import format_exc

import numpy
import pandas

from superduper import logging
from superduper.base import Base
from superduper.base.metadata import JOB_PHASE_FAILED

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class Signal(Base):
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


class Change(Base):
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


class Create(Base):
    """
    Class for component creation events.

    :param context: the component context of creation.
    :param path: path of the component to be created
    :param data: the data of the component
    :param parent: the parent of the component (if any)
    :param children: the children of the component (if any)
    """

    queue: t.ClassVar[str] = '_apply'

    context: str
    path: str
    data: t.Dict
    parent: list | None = None
    children: t.List | None = None

    @property
    def component(self):
        return self.path.split('.')[-1]

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

            component = db.load(component=self.component, uuid=self.data['uuid'])

            if self.children:
                for child in self.children:
                    db.metadata.create_parent_child(
                        child_component=child[0],
                        child_identifier=child[1],
                        child_uuid=child[2],
                        parent_component=self.component,
                        parent_identifier=component.identifier,
                        parent_uuid=component.uuid,
                    )

            component.on_create()
            logging.info(
                f'Created {self.path.split("/")[-1]}:'
                f'{self.data["identifier"]}:{self.data["uuid"]}'
            )

        except Exception as e:
            db.metadata.set_component_status(
                component=self.component,
                uuid=self.data['uuid'],
                status_update={
                    'phase': JOB_PHASE_FAILED,
                    'reason': f'Failed to create: {str(e)}',
                    'message': format_exc(),
                },
            )
            raise e

    @property
    def huuid(self):
        """Return the hashed uuid."""
        return f'{self.component}:' f'{self.data["identifier"]}:' f'{self.data["uuid"]}'


class Delete(Base):
    """
    Class for component deletion events.

    :param component: the type of component to be created
    :param identifier: the identifier of the component to be deleted
    """

    queue: t.ClassVar[str] = '_apply'

    component: str
    identifier: str

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

            object.cleanup()

        except Exception as e:
            db.metadata.set_component_status(
                component=self.component,
                uuid=self.identifier,
                status_update={
                    'phase': JOB_PHASE_FAILED,
                    'reason': f'Failed to delete: {str(e)}',
                    'message': format_exc(),
                },
            )
            raise e


class Update(Base):
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
                status_update={
                    'phase': JOB_PHASE_FAILED,
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
