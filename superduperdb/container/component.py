"""
The component module provides the base class for all components in SuperDuperDB.
"""


from __future__ import annotations

import dataclasses as dc
import typing as t

from superduperdb.container.job import ComponentJob, Job
from superduperdb.container.serializable import Serializable

if t.TYPE_CHECKING:
    from superduperdb.container.dataset import Dataset
    from superduperdb.db.base.db import DB


class Component(Serializable):
    """
    Base component which model, listeners, learning tasks etc. inherit from.
    """

    type_id: t.ClassVar[str]

    if t.TYPE_CHECKING:
        identifier: t.Optional[str]
        version: t.Optional[int]

    def on_create(self, db: DB) -> None:
        """Called the first time this component is created

        :param db: the db that creates the component
        """
        pass

    def on_load(self, db: DB) -> None:
        """Called when this component is loaded from the data store

        :param db: the db that loaded the component
        """
        pass

    @property
    def child_components(self) -> t.Sequence[t.Any]:
        return []

    @property
    def unique_id(self) -> str:
        if getattr(self, 'version', None) is None:
            raise Exception('Version not yet set for component uniqueness')
        return f'{self.type_id}/' f'{self.identifier}/' f'{self.version}'

    def dict(self) -> t.Dict[str, t.Any]:
        return dc.asdict(self)

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
                'distributed': False,
                'validation_set': validation_set,
                'metrics': metrics,
            },
        )

    def schedule_jobs(
        self,
        database: DB,
        dependencies: t.Sequence[Job] = (),
        distributed: bool = False,
        verbose: bool = False,
    ) -> t.Sequence[t.Any]:
        """Run the job for this listener

        :param database: The db to process
        :param dependencies: A sequence of dependencies,
        :param distributed: Is the computation distributed
        :param verbose: If true, print more information
        """
        return []

    @classmethod
    def make_unique_id(cls, type_id: str, identifier: str, version: int) -> str:
        return f'{type_id}/{identifier}/{version}'
