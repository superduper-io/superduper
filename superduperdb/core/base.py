# ruff: noqa: F821
import io
import typing as t

from dask.distributed import Future

from superduperdb.core.job import ComponentJob
import dataclasses as dc

from superduperdb.core.serializable import Serializable
from superduperdb.datalayer.base.artifacts import ArtifactStore
from superduperdb.misc.serialization import serializers


class ArtifactSavingError(Exception):
    pass


class Artifact:
    def __init__(
        self,
        _artifact: t.Optional[t.Any] = None,
        serializer: str = 'pickle',
        info: t.Optional[t.Dict] = None,
        file_id: t.Optional[t.Any] = None,
    ):
        self.serializer = serializer
        self._artifact = _artifact
        self.info = info
        self.file_id = file_id

    def __repr__(self):
        return f'<Artifact artifact={str(self._artifact)} serializer={self.serializer}>'

    def save(self, artifact_store: ArtifactStore, cache, replace=False):
        object_id = id(self._artifact)
        if object_id not in cache:
            file_id, sha1 = artifact_store.create_artifact(
                object=self._artifact, serializer=self.serializer, info=self.info
            )
            if replace and self.file_id is not None:
                artifact_store.delete_artifact(self.file_id)
            elif not replace and self.file_id is not None:
                raise ArtifactSavingError(
                    "Something has gone wrong in saving, "
                    f"Artifact {self._artifact} was already saved."
                )
            self.file_id = file_id
            details = {
                'file_id': file_id,
                'sha1': sha1,
                'id': object_id,
                'serializer': self.serializer,
                'info': self.info,
            }
            cache[object_id] = details
        return cache[id(self._artifact)]

    @staticmethod
    def load(r, artifact_store: ArtifactStore, cache):
        artifact = artifact_store.load_artifact(
            r['file_id'], r['serializer'], info=r['info']
        )
        if r['file_id'] in cache:
            return cache[r['file_id']]
        a = Artifact(
            _artifact=artifact,
            serializer=r['serializer'],
            info=r['info'],
        )
        cache[r['file_id']] = a._artifact
        return a

    def dict(self):
        ...

    @property
    def a(self):
        return self._artifact

    def serialize(self):
        if self.save_method is not None:
            f = io.BytesIO()
            getattr(self._artifact, self.save_method)(f)
        return serializers[self.serializer].encode(self._artifact)


@dc.dataclass
class Component(Serializable):
    """
    Base component which models, watchers, learning tasks etc. inherit from.

    :param identifier: Unique ID
    """

    variety: t.ClassVar[str]

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
        metrics: t.List[str],
    ):
        return ComponentJob(
            component_identifier=self.identifier,
            method_name='predict',
            variety='model',
            kwargs={
                'remote': False,
                'validation_set': validation_set,
                'metrics': metrics,
            },
        )

    # ruff: noqa: E501
    def _validate(
        self,
        db: 'superduperdb.datalayer.base.database.Database',  # type: ignore[name-defined]
        validation_set: t.Union[str, 'Dataset'],  # type: ignore[name-defined]
        metrics: t.List[t.Union['Metric', str]],  # type: ignore[name-defined]
    ):
        raise NotImplementedError

    # ruff: noqa: E501
    def validate(
        self,
        db: 'superduperdb.datalayer.base.database.Database',  # type: ignore[name-defined]
        validation_set: t.Union[str, 'Dataset'],  # type: ignore[name-defined]
        metrics: t.List[t.Union['Metric', str]],  # type: ignore[name-defined]
        remote: bool = False,
        dependencies: t.Sequence[Future] = (),
    ):
        from .dataset import Dataset
        from .metric import Metric

        db.add(self)

        if isinstance(validation_set, Dataset):
            db.add(validation_set)
            validation_set = validation_set.identifier

        for i, m in enumerate(metrics):
            if isinstance(m, Metric):
                db.add(m)
                metrics[i] = m.identifier

        if remote:
            return self.create_validation_job(
                validation_set=validation_set,
                metrics=metrics,
            )(db=db, remote=True, dependencies=dependencies)

        output = self._validate(
            db=db,
            validation_set=validation_set,
            metrics=metrics,
        )
        if validation_set not in self.metric_values:
            self.metric_values[validation_set] = {}
        if self.metric_values[validation_set]:
            self.metric_values[validation_set].update(output)
        else:
            self.metric_values[validation_set] = output
        db.metadata.update_object(
            variety=self.variety,
            identifier=self.identifier,
            key='metric_values',
            value=self.metric_values,
        )
        return self

    def schedule_jobs(self, database, dependencies=()):
        return []

    @classmethod
    def make_unique_id(cls, variety, identifier, version):
        return f'{variety}/{identifier}/{version}'
