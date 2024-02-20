import dataclasses as dc
import typing as t

import numpy
from sklearn.base import BaseEstimator
from tqdm import tqdm

from superduperdb.backends.base.query import Select
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.base.datalayer import Datalayer
from superduperdb.components.datatype import DataType, pickle_serializer
from superduperdb.components.metric import Metric
from superduperdb.components.model import (
    Mapping,
    _Fittable,
    _Predictor,
    _TrainingConfiguration,
)
from superduperdb.jobs.job import Job


def _get_data_from_query(
    select: Select,
    X: str,
    db: Datalayer,
    y: t.Optional[str] = None,
    y_preprocess: t.Optional[t.Callable] = None,
    preprocess: t.Optional[t.Callable] = None,
):
    if y is None:
        data = QueryDataset(
            select=select,
            mapping=Mapping([X], signature='singleton'),
            transform=preprocess,
            db=db,
        )
    else:
        y_preprocess = y_preprocess or (lambda x: x)
        preprocess = preprocess or (lambda x: x)
        data = QueryDataset(
            select=select,
            mapping=Mapping([X, y], signature='*args'),
            transform=lambda x, y: (preprocess(x), y_preprocess(y)),
            db=db,
        )

    rows = []
    for i in tqdm(range(len(data))):
        rows.append(data[i])
    if y is not None:
        X_arr = [x[0] for x in rows]
        y_arr = [x[1] for x in rows]
    else:
        X_arr = rows

    if isinstance(X[0], numpy.ndarray):
        X_arr = numpy.stack(X_arr)
    if y is not None:
        y_arr = [r[1] for r in rows]
        if isinstance(y_arr[0], numpy.ndarray):
            y_arr = numpy.stack(y_arr)
        return X_arr, y_arr
    return X_arr, None


@dc.dataclass
class SklearnTrainingConfiguration(_TrainingConfiguration):
    fit_params: t.Dict = dc.field(default_factory=dict)
    predict_params: t.Dict = dc.field(default_factory=dict)
    y_preprocess: t.Optional[t.Callable] = None


@dc.dataclass(kw_only=True)
class Estimator(_Predictor, _Fittable):
    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, DataType]]] = (
        ('object', pickle_serializer),
    )
    signature: t.ClassVar[str] = 'singleton'
    object: BaseEstimator
    preprocess: t.Optional[t.Callable] = None
    postprocess: t.Optional[t.Callable] = None

    def schedule_jobs(
        self,
        db: Datalayer,
        dependencies: t.Sequence[Job] = (),
        verbose: bool = False,
    ) -> t.Sequence[t.Any]:
        jobs = _Fittable.schedule_jobs(self, db, dependencies=dependencies)
        jobs.extend(
            _Predictor.schedule_jobs(self, db, dependencies=[*dependencies, *jobs])
        )
        return jobs

    def __getattr__(self, item):
        if item in ['transform', 'predict_proba', 'score']:
            return getattr(self.object, item)
        else:
            return super().__getattribute__(item)

    def predict_one(self, X):
        X = X[None, :]
        if self.preprocess is not None:
            X = self.preprocess(X)
        X = self.object.predict(X, **self.predict_kwargs)[0]
        if self.postprocess is not None:
            X = self.postprocess(X)
        return X

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        if self.preprocess is not None:
            inputs = []
            for i in range(len(dataset)):
                args, kwargs = dataset[i]
                inputs.append(self.preprocess(*args, **kwargs))
            dataset = inputs
        else:
            dataset = [dataset[i] for i in range(len(dataset))]
        out = self.object.predict(dataset, **self.predict_kwargs)
        if self.postprocess is not None:
            out = list(out)
            out = list(map(self.postprocess, out))
        return out

    def _fit(  # type: ignore[override]
        self,
        X,
        y=None,
        configuration: t.Optional[SklearnTrainingConfiguration] = None,
        data_prefetch: bool = False,
        db: t.Optional[Datalayer] = None,
        metrics: t.Optional[t.Sequence[Metric]] = None,
        select: t.Optional[Select] = None,
        validation_sets: t.Optional[t.Sequence[str]] = None,
    ):
        if configuration is not None:
            self.training_configuration = configuration
        if select is not None:
            self.training_select = select
        if isinstance(X, str):
            self.train_X = X
            self.train_y = y
        if isinstance(X, str):
            if isinstance(self.training_configuration, str):
                raise TypeError('self.training_configuration cannot be str')
            y_preprocess = None
            if self.training_configuration is not None:
                y_preprocess = self.training_configuration.get('y_preprocess', None)
            if select is None or db is None:
                raise ValueError('Neither select nor db can be None')

            X, y = _get_data_from_query(
                select=select,
                db=db,
                X=X,
                y=y,
                preprocess=self.preprocess,
                y_preprocess=y_preprocess,
            )
        if self.training_configuration is not None:
            assert not isinstance(self.training_configuration, str)
            to_return = self.object.fit(
                X,
                y,
                **self.training_configuration.get('fit_params', {}),
            )
        else:
            to_return = self.object.fit(X, y)

        if validation_sets is not None:
            assert db is not None
            assert metrics is not None
            results = {}
            for validation_set in validation_sets:
                results.update(
                    # TODO: this call could never work
                    self._validate(
                        db=db,
                        validation_set=validation_set,
                        metrics=metrics,
                    )
                )
            self.append_metrics(results)

        if db is not None:
            db.replace(self, upsert=True)
        return to_return
