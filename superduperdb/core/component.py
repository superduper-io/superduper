"""
The component module provides the base class for all components in SuperDuperDB.
"""


# ruff: noqa: F821
from __future__ import annotations

import dataclasses as dc
import typing as t

from superduperdb.core.job import ComponentJob, Job
from superduperdb.core.serializable import Serializable

if t.TYPE_CHECKING:
    from superduperdb.datalayer.base.datalayer import Datalayer
    from superduperdb.datalayer.base.dataset import Dataset


@dc.dataclass
class Component(Serializable):
    """
    Base component which models, watchers, learning tasks etc. inherit from.

    :param identifier: Unique ID
    """

    variety: t.ClassVar[str]

    def on_create(self, db: Datalayer) -> None:
        """Called the first time this component is created

        :param db: the datalayer that created the component
        """
        pass

    def on_load(self, db: Datalayer) -> None:
        """Called when this component is loaded from the data store

        :param db: the datalayer that loaded the component
        """
        pass

    @property
    def child_components(self) -> t.Sequence[t.Any]:
        return []

    @property
    def unique_id(self) -> str:
        if self.version is None:
            raise Exception('Version not yet set for component uniqueness')
        return f'{self.variety}/{self.identifier}/{self.version}'

    def dict(self) -> t.Dict[str, t.Any]:
        return dc.asdict(self)

    def create_validation_job(
        self,
        validation_set: t.Union[str, Dataset],
        metrics: t.Sequence[str],
    ) -> ComponentJob:
        return ComponentJob(
            component_identifier=self.identifier,
            method_name='predict',
            variety='model',
            kwargs={
                'distributed': False,
                'validation_set': validation_set,
                'metrics': metrics,
            },
        )

    def schedule_jobs(
        self,
        database: Datalayer,
        dependencies: t.Sequence[Job] = (),
        distributed: bool = False,
        verbose: bool = False,
    ) -> t.Sequence[t.Any]:
        """Run the job for this watcher

        :param database: The datalayer to process
        :param dependencies: A sequence of dependencies,
        :param distributed: Is the computation distributed
        :param verbose: If true, print more information
        """
        return []

    @classmethod
    def make_unique_id(cls, variety: str, identifier: str, version: int) -> str:
        return f'{variety}/{identifier}/{version}'
