"""
The component module provides the base class for all components in SuperDuperDB.
"""


from __future__ import annotations

import typing as t

from superduperdb.backends.base.artifact import ArtifactStore
from superduperdb.base.serializable import Serializable
from superduperdb.jobs.job import ComponentJob, Job

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.dataset import Dataset


class Component(Serializable):
    """
    Base component which model, listeners, learning tasks etc. inherit from.
    """

    type_id: t.ClassVar[str]

    if t.TYPE_CHECKING:
        identifier: t.Optional[str]
        version: t.Optional[int]

    def pre_create(self, db: Datalayer) -> None:
        """Called the first time this component is created

        :param db: the db that creates the component
        """
        assert db

    def post_create(self, db: Datalayer) -> None:
        """Called after the first time this component is created.
        Generally used if ``self.version`` is important in this logic.

        :param db: the db that creates the component
        """
        assert db
        assert db

    def on_load(self, db: Datalayer) -> None:
        """Called when this component is loaded from the data store

        :param db: the db that loaded the component
        """
        assert db

    @property
    def child_components(self) -> t.Sequence[t.Any]:
        return []

    @property
    def serialized(self):
        serialized = self.serialize()
        artifacts = tuple(set(ArtifactStore.get_artifacts(serialized)))
        return serialized, artifacts

    @property
    def unique_id(self) -> str:
        if getattr(self, 'version', None) is None:
            raise Exception('Version not yet set for component uniqueness')
        return f'{self.type_id}/' f'{self.identifier}/' f'{self.version}'

    def create_validation_job(
        self,
        validation_set: t.Union[str, Dataset],
        metrics: t.Sequence[str],
    ) -> ComponentJob:
        assert self.identifier is not None
        return ComponentJob(
            component_identifier=self.identifier,
            method_name='predict',
            type_id='model',
            kwargs={
                'validation_set': validation_set,
                'metrics': metrics,
            },
        )

    def schedule_jobs(
        self,
        database: Datalayer,
        dependencies: t.Sequence[Job] = (),
        verbose: bool = False,
    ) -> t.Sequence[t.Any]:
        """Run the job for this listener

        :param database: The db to process
        :param dependencies: A sequence of dependencies,
        :param verbose: If true, print more information
        """
        return []

    @classmethod
    def make_unique_id(cls, type_id: str, identifier: str, version: int) -> str:
        return f'{type_id}/{identifier}/{version}'
