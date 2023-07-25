# ruff: noqa: F821
from __future__ import annotations
import typing as t
from superduperdb.core.job import ComponentJob
from superduperdb.core.serializable import Serializable
import dataclasses as dc

if t.TYPE_CHECKING:
    from superduperdb.datalayer.base.datalayer import Datalayer


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
    def child_components(self):
        return []

    @property
    def unique_id(self):
        if self.version is None:
            raise Exception('Version not yet set for component uniqueness')
        return f'{self.variety}/{self.identifier}/{self.version}'

    def dict(self):
        return dc.asdict(self)

    def create_validation_job(
        self,
        validation_set: t.Union[str, 'Dataset'],  # type: ignore[name-defined]
        metrics: t.Sequence[str],
    ):
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

    def schedule_jobs(self, database, dependencies=()):
        return []

    @classmethod
    def make_unique_id(cls, variety, identifier, version):
        return f'{variety}/{identifier}/{version}'
