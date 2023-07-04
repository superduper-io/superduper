from dask.distributed import Future
import typing as t

from superduperdb.core.dataset import Dataset
from superduperdb.core.job import Job, ComponentJob
from superduperdb.core.metric import Metric
from superduperdb.core.base import Component, Placeholder
from superduperdb.core.encoder import Encoder
from superduperdb.datalayer.base.query import Select
from superduperdb.misc.special_dicts import MongoStyleDict
from superduperdb.queries.serialization import to_dict, from_dict

EncoderArg = t.Union[Encoder, Placeholder, None, str]


class TrainingConfiguration(Component):
    """
    Training configuration object, containing all settings necessary for a particular
    learning-task use-case to be serialized and initiated. The object is ``callable``
    and returns a class which may be invoked to apply training.

    :param identifier: Unique identifier of configuration
    :param **parameters: Key-values pairs, the variables which configure training.
    """

    variety = 'training_configuration'

    def __init__(self, identifier, **parameters):
        super().__init__(identifier)
        for k, v in parameters.items():
            setattr(self, k, v)

    def get(self, k, default=None):
        return getattr(self, k, default)


class Model(Component):
    """
    Model component which wraps a model to become serializable

    :param object: Model object, e.g. sklearn model, etc..
    :param identifier: Unique identifying ID
    :param encoder: Encoder instance (optional)
    """

    variety: str = 'model'
    object: t.Any
    identifier: str
    encoder: EncoderArg

    def __init__(
        self,
        object: t.Any,
        identifier: str,
        encoder: EncoderArg = None,
        training_configuration: t.Optional[TrainingConfiguration] = None,
        training_select: t.Optional[Select] = None,
        train_X: t.Optional[t.Union[str, t.List[str]]] = None,
        train_y: t.Optional[t.Union[str, t.List[str]]] = None,
        metrics: t.Optional[t.List[Metric]] = None,
    ):
        super().__init__(identifier)
        self.object = object

        if isinstance(encoder, str):
            self.encoder: EncoderArg = Placeholder(encoder, 'type')
        else:
            self.encoder: EncoderArg = encoder  # type: ignore

        try:
            self.predict_one = object.predict_one
        except AttributeError:
            pass

        if not hasattr(self, '_predict'):
            try:
                self._predict = object._predict
            except AttributeError:
                pass

        self.training_configuration = training_configuration
        self.training_select = training_select
        self.train_X = train_X
        self.train_y = train_y
        self.metrics = metrics
        self.metric_values: t.Dict = {}
        self.future: t.Optional[Future] = None

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

    def asdict(self):
        return {
            'identifier': self.identifier,
            'type': None if self.encoder is None else self.encoder.identifier,
        }

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
                'select': to_dict(select) if select else None,
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
                'select': to_dict(select) if select else None,
                **kwargs,
            },
        )

    def _fit(
        self,
        X: t.Any,
        y: t.Optional[t.Any] = None,
        db: t.Optional['BaseDatabase'] = None,  # type: ignore[name-defined]
        select: t.Optional[Select] = None,
        dependencies: t.List[Job] = (),  # type: ignore[assignment]
        configuration: t.Optional[TrainingConfiguration] = None,
        validation_sets: t.Optional[t.List[t.Union[str, Dataset]]] = None,
        metrics: t.Optional[t.List[Metric]] = None,
    ):
        raise NotImplementedError

    # ruff: noqa: F821
    def fit(
        self,
        X: t.Any,
        y: t.Optional[t.Any] = None,
        db: t.Optional['BaseDatabase'] = None,  # type: ignore[name-defined]
        select: t.Optional[Select] = None,
        remote: bool = False,
        dependencies: t.List[Job] = (),  # type: ignore[assignment]
        configuration: t.Optional[TrainingConfiguration] = None,
        validation_sets: t.Optional[t.List[t.Union[str, Dataset]]] = None,
        metrics: t.Optional[t.List[Metric]] = None,
        **kwargs,
    ):
        if isinstance(select, dict):
            select = from_dict(select)

        if validation_sets:
            validation_sets = list(validation_sets)  # type: ignore[arg-type]
            for i, vs in enumerate(validation_sets):
                if isinstance(vs, Dataset):
                    db.add(vs)  # type: ignore[union-attr]
                    validation_sets[i] = vs.identifier

        if db is not None:
            db.add(self)

        if remote:
            return self.create_fit_job(
                X,
                select=select,
                y=y,
                **kwargs,
            )(db=db, remote=remote, dependencies=dependencies)
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

    # ruff: noqa: F821
    def predict(
        self,
        X: t.Any,
        db: 'BaseDatabase' = None,  # type: ignore[name-defined]
        select: t.Optional[Select] = None,
        remote: bool = False,
        ids: t.Optional[t.List[str]] = None,
        max_chunk_size: t.Optional[int] = None,
        dependencies: t.List[Job] = (),  # type: ignore[assignment]
        watch: bool = False,
        **kwargs,
    ):
        if isinstance(select, dict):
            select = from_dict(select)

        if watch:
            from superduperdb.core.watcher import Watcher

            return db.add(
                Watcher(
                    model=self,
                    select=select,  # type: ignore[arg-type]
                    key=X,
                ),
                dependencies=dependencies,
            )

        if db is not None:
            db.add(self)

        if remote:
            return self.create_predict_job(
                X, select=select, ids=ids, max_chunk_size=max_chunk_size, **kwargs
            )(db=db, remote=remote, dependencies=dependencies)
        else:
            if select is not None and ids is None:
                ids = [
                    str(r[db.databackend.id_field])
                    for r in db.execute(select.select_ids)
                ]
                return self.predict(
                    X=X,
                    db=db,
                    ids=ids,
                    remote=False,
                    select=select,
                    max_chunk_size=max_chunk_size,
                    **kwargs,
                )

            elif select is not None and ids is not None and max_chunk_size is None:
                docs = list(db.execute(select.select_using_ids(ids)))
                if X != '_base':
                    X_data = [MongoStyleDict(r.unpack())[X] for r in docs]
                else:
                    X_data = [r.unpack() for r in docs]
                outputs = self.predict(X=X_data, remote=False)
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
                for i in range(0, len(ids), max_chunk_size):
                    self.predict(
                        X=X,
                        db=db,
                        ids=ids[i : i + max_chunk_size],
                        select=select,
                        max_chunk_size=None,
                        **kwargs,
                    )
                return

            else:
                return self._predict(X, **kwargs)


class ModelEnsemblePredictionError(Exception):
    pass


# TODO make Component less dogmatic about having just one ``self.object`` type thing
class ModelEnsemble:
    variety: str = 'model'

    def __init__(self, models: t.List[t.Union[Model, str]]):
        self._model_ids = []
        for m in models:
            if isinstance(m, Model):
                setattr(self, m.identifier, m)
                self._model_ids.append(m.identifier)
            elif isinstance(m, str):
                setattr(self, m, Placeholder('model', m))
                self._model_ids.append(m)

    def __getitem__(self, submodel: t.Union[int, str]):
        if isinstance(submodel, int):
            submodel = next(m for i, m in enumerate(self._model_ids) if i == submodel)
        submodel = getattr(self, submodel)
        assert isinstance(submodel, Model)
        return submodel
