from dask.distributed import Future
import inspect
import dataclasses as dc
import typing as t

from superduperdb.core.dataset import Dataset
from superduperdb.core.job import Job, ComponentJob
from superduperdb.core.metric import Metric
from superduperdb.core.artifact import Artifact
from superduperdb.core.component import Component
from superduperdb.core.encoder import Encoder
from superduperdb.core.serializable import Serializable
from superduperdb.datalayer.base.query import Select
from superduperdb.misc.special_dicts import MongoStyleDict

EncoderArg = t.Union[Encoder, str, None]
ObjectsArg = t.List[t.Union[t.Any, Artifact]]
DataArg = t.Optional[t.Union[str, t.List[str]]]


def TrainingConfiguration(
    identifier: str,
    **kwargs,
):
    return _TrainingConfiguration(identifier=identifier, kwargs=kwargs)


@dc.dataclass
class _TrainingConfiguration(Component):
    """
    Training configuration object, containing all settings necessary for a particular
    learning-task use-case to be serialized and initiated. The object is ``callable``
    and returns a class which may be invoked to apply training.

    :param identifier: Unique identifier of configuration
    :param **parameters: Key-values pairs, the variables which configure training.
    """

    variety: t.ClassVar[str] = 'training_configuration'

    identifier: str
    kwargs: t.Optional[t.Dict] = None
    version: t.Optional[int] = None

    def get(self, k, default=None):
        try:
            return getattr(self, k)
        except AttributeError:
            return self.kwargs.get(k, default=default)


class PredictMixin:
    # ruff: noqa: F821
    def predict(
        self,
        X: t.Any,
        db: 'BaseDatabase' = None,  # type: ignore[name-defined]
        select: t.Optional[Select] = None,
        distributed: bool = False,
        ids: t.Optional[t.List[str]] = None,
        max_chunk_size: t.Optional[int] = None,
        dependencies: t.List[Job] = (),  # type: ignore[assignment]
        watch: bool = False,
        one: bool = False,
        **kwargs,
    ):
        if isinstance(select, dict):
            select = Serializable.deserialize(select)

        if watch:
            from superduperdb.core.watcher import Watcher

            return db.add(
                Watcher(
                    model=self,  # type: ignore[arg-type]
                    select=select,  # type: ignore[arg-type]
                    key=X,
                ),
                dependencies=dependencies,
            )

        if db is not None:
            db.add(self)

        if distributed:
            assert not one
            return self.create_predict_job(
                X, select=select, ids=ids, max_chunk_size=max_chunk_size, **kwargs
            )(db=db, distributed=distributed, dependencies=dependencies)
        else:
            if select is not None and ids is None:
                assert not one
                ids = [
                    str(r[db.databackend.id_field])
                    for r in db.execute(select.select_ids)
                ]
                return self.predict(
                    X=X,
                    db=db,
                    ids=ids,
                    distributed=False,
                    select=select,
                    max_chunk_size=max_chunk_size,
                    one=False,
                    **kwargs,
                )

            elif select is not None and ids is not None and max_chunk_size is None:
                assert not one
                docs = list(db.execute(select.select_using_ids(ids)))
                if X != '_base':
                    X_data = [MongoStyleDict(r.unpack())[X] for r in docs]
                else:
                    X_data = [r.unpack() for r in docs]

                outputs = self.predict(X=X_data, one=False, distributed=False)
                if self.encoder is not None:
                    # ruff: noqa: E501
                    outputs = [self.encoder(x).encode() for x in outputs]  # type: ignore[operator]

                select.model_update(
                    db=db,
                    model=self.identifier,
                    outputs=outputs,
                    key=X,
                    ids=ids,
                )
                return

            elif select is not None and ids is not None and max_chunk_size is not None:
                assert not one
                for i in range(0, len(ids), max_chunk_size):
                    self.predict(
                        X=X,
                        db=db,
                        ids=ids[i : i + max_chunk_size],
                        select=select,
                        max_chunk_size=None,
                        one=False,
                        **kwargs,
                    )
                return

            else:
                if 'one' in inspect.signature(self._predict).parameters:
                    return self._predict(X, one=one, **kwargs)
                else:
                    return self._predict(X, **kwargs)


@dc.dataclass
class Model(Component, PredictMixin):
    """
    Model component which wraps a model to become serializable

    :param object: Model object, e.g. sklearn model, etc..
    :param encoder: Encoder instance (optional)
    :param variety: ...
    """

    variety: t.ClassVar[str] = 'model'
    artifacts: t.ClassVar[t.List[str]] = ['object']

    identifier: str
    object: t.Union[Artifact, t.Any]
    encoder: EncoderArg = None
    train_X: DataArg = None
    train_y: DataArg = None
    training_select: t.Union[Select, None] = None
    metrics: t.List[t.Union[str, Metric, None]] = None
    training_configuration: t.Union[str, _TrainingConfiguration, None] = None
    version: t.Optional[int] = None
    metric_values: t.Optional[t.Dict] = dc.field(default_factory=dict)
    future: t.Optional[Future] = None
    device: str = "cpu"

    def __post_init__(self):
        if not isinstance(self.object, Artifact):
            self.object = Artifact(artifact=self.object)

    @property
    def child_components(self) -> t.List[t.Tuple[str, str]]:
        out = []
        if self.encoder is not None:
            out.append(('encoder', 'encoder'))
        if self.training_configuration is not None:
            out.append(('training_configuration', 'training_configuration'))
        return out

    @property
    def training_keys(self):
        out = []
        if isinstance(self.train_X, list):
            out.extend(self.train_X)
        elif isinstance(self.train_X, str):
            out.append(self.train_X)
        if isinstance(self.train_y, list):
            out.extend(self.train_y)
        elif isinstance(self.train_y, str):
            out.append(self.train_y)
        return out

    def append_metrics(self, d):
        for k in d:
            if k not in self.metric_values:
                self.metric_values[k] = []
            self.metric_values[k].append(d[k])

    def create_predict_job(
        self,
        X: str,
        select: t.Optional[Select] = None,
        ids: t.Optional[t.List[str]] = None,
        max_chunk_size: t.Optional[int] = None,
        **kwargs,
    ):
        return ComponentJob(
            component_identifier=self.identifier,
            method_name='predict',
            variety='model',
            args=[X],
            kwargs={
                'remote': False,
                'select': select.serialize() if select else None,
                'ids': ids,
                'max_chunk_size': max_chunk_size,
                **kwargs,
            },
        )

    def create_fit_job(
        self,
        X: t.Union[str, t.List[str]],
        select: t.Optional[Select] = None,
        y: t.Optional[str] = None,
        **kwargs,
    ):
        return ComponentJob(
            component_identifier=self.identifier,
            method_name='fit',
            variety='model',
            args=[X],
            kwargs={
                'y': y,
                'remote': False,
                'select': select.serialize() if select else None,
                **kwargs,
            },
        )

    def _fit(
        self,
        X: t.Any,
        y: t.Any = None,
        db: t.Optional['Datalayer'] = None,  # type: ignore[name-defined]
        select: t.Optional[Select] = None,
        dependencies: t.List[Job] = (),  # type: ignore[assignment]
        configuration: t.Optional[_TrainingConfiguration] = None,
        validation_sets: t.Optional[t.List[t.Union[str, Dataset]]] = None,
        metrics: t.Optional[t.List[Metric]] = None,
    ):
        raise NotImplementedError

    # ruff: noqa: F821
    def fit(
        self,
        X: t.Any,
        y: t.Any = None,
        db: t.Optional['Datalayer'] = None,  # type: ignore[name-defined]
        select: t.Optional[Select] = None,
        distributed: bool = False,
        dependencies: t.List[Job] = (),  # type: ignore[assignment]
        configuration: t.Optional[_TrainingConfiguration] = None,
        validation_sets: t.Optional[t.List[t.Union[str, Dataset]]] = None,
        metrics: t.Optional[t.List[Metric]] = None,
        data_prefetch: bool = False,
        **kwargs,
    ):
        if isinstance(select, dict):
            select = Serializable.deserialize(select)

        if validation_sets:
            validation_sets = list(validation_sets)  # type: ignore[arg-type]
            for i, vs in enumerate(validation_sets):
                if isinstance(vs, Dataset):
                    db.add(vs)  # type: ignore[union-attr]
                    validation_sets[i] = vs.identifier

        if db is not None:
            db.add(self)

        if distributed:
            return self.create_fit_job(
                X,
                select=select,
                y=y,
                **kwargs,
            )(db=db, distributed=True, dependencies=dependencies)
        else:
            return self._fit(
                X,
                y=y,
                db=db,
                validation_sets=validation_sets,
                metrics=metrics,
                configuration=configuration,
                select=select,
                data_prefetch=data_prefetch,
                **kwargs,
            )


class ModelEnsemblePredictionError(Exception):
    pass


@dc.dataclass
class ModelEnsemble(Component):
    variety: t.ClassVar[str] = 'model'

    identifier: str
    models: t.List[t.Union[str, Model]]
    version: t.Optional[int] = None
    db: dc.InitVar[t.Any] = None
    train_X: t.Optional[t.List[str]] = None
    train_y: t.Optional[t.Union[t.List[str], str]] = None
    metric_values: t.Optional[t.Dict] = dc.field(default_factory=dict)

    def __post_init__(self, db):
        for i, m in enumerate(self.models):
            if isinstance(m, str):
                self.models[i] = db.load('model', m)

    def __getitem__(self, submodel: t.Union[int, str]):
        if isinstance(submodel, int):
            submodel = next(m for i, m in enumerate(self._model_ids) if i == submodel)
        else:
            submodel = getattr(self, submodel)

        if isinstance(submodel, Model):
            return submodel
        raise ValueError(f'Expected a Model but got {type(submodel)}: {submodel}')

    @property
    def training_keys(self):
        out = []
        out.extend(self.train_X)
        if isinstance(self.train_y, list):
            out.extend(self.train_y)
        elif isinstance(self.train_y, str):
            out.append(self.train_y)
        return out

    # ruff: noqa: F821
    def fit(
        self,
        X: t.Any,
        y: t.Any = None,
        db: t.Optional['Datalayer'] = None,  # type: ignore[name-defined]
        select: t.Optional[Select] = None,
        distributed: bool = False,
        dependencies: t.List[Job] = (),  # type: ignore[assignment]
        configuration: t.Optional[_TrainingConfiguration] = None,
        validation_sets: t.Optional[t.List[t.Union[str, Dataset]]] = None,
        metrics: t.Optional[t.List[Metric]] = None,
        **kwargs,
    ):
        if isinstance(select, dict):
            select = Serializable.deserialize(select)

        if validation_sets:
            validation_sets = list(validation_sets)  # type: ignore[arg-type]
            for i, vs in enumerate(validation_sets):
                if isinstance(vs, Dataset):
                    db.add(vs)  # type: ignore[union-attr]
                    validation_sets[i] = vs.identifier

        if db is not None:
            db.add(self)

        if distributed:
            return self.create_fit_job(
                X,
                select=select,
                y=y,
                **kwargs,
            )(db=db, distributed=True, dependencies=dependencies)
        else:
            return self._fit(
                X,
                y=y,
                db=db,
                validation_sets=validation_sets,
                metrics=metrics,
                configuration=configuration,
                select=select,
                **kwargs,
            )

    def append_metrics(self, d):
        for k in d:
            if k not in self.metric_values:
                self.metric_values[k] = []
            self.metric_values[k].append(d[k])
