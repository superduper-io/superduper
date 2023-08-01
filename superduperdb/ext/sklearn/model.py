import dataclasses as dc
import typing as t

import numpy
from tqdm import tqdm

from superduperdb.container.artifact import Artifact
from superduperdb.container.metric import Metric
from superduperdb.container.model import Model, _TrainingConfiguration
from superduperdb.db.base.db import DB
from superduperdb.db.base.query import Select
from superduperdb.db.query_dataset import QueryDataset


def get_data_from_query(
    select: Select,
    X: str,
    db: DB,
    y: t.Optional[str] = None,
    y_preprocess: t.Optional[t.Callable] = None,
    preprocess: t.Optional[t.Callable] = None,
):
    def transform(r):
        out = {}
        if X == '_base':
            out.update(**preprocess(r))
        else:
            out[X] = preprocess(r[X])
        if y is not None:
            out[y] = y_preprocess(r[y]) if y_preprocess else r[y]
        return out

    data = QueryDataset(
        select=select,
        keys=[X] if y is None else [X, y],
        transform=transform,
        db=db,
    )
    documents = []
    for i in tqdm(range(len(data))):
        r = data[i]
        documents.append(r)
    X_arr = [r[X] for r in documents]
    if isinstance(X[0], numpy.ndarray):
        X_arr = numpy.stack(X_arr)
    if y is not None:
        y_arr = [r[y] for r in documents]
        if y_preprocess is not None:
            y_arr = [y_preprocess(item) for item in y_arr]
        if isinstance(y[0], numpy.ndarray):
            y_arr = numpy.stack(y_arr)
    return X_arr, y_arr


@dc.dataclass
class SklearnTrainingConfiguration(_TrainingConfiguration):
    fit_params: t.Dict = dc.field(default_factory=dict)
    predict_params: t.Dict = dc.field(default_factory=dict)
    y_preprocess: t.Optional[Artifact] = None


@dc.dataclass
class Estimator(Model):
    def __post_init__(self):
        if self.predict_method is not None:
            assert self.predict_method == 'predict'
        self.predict_method = 'predict'
        super().__post_init__()

    @property
    def estimator(self):
        return self.object.artifact

    def __getattr__(self, item):
        if item in ['transform', 'predict_proba', 'score']:
            return getattr(self.estimator, item)
        else:
            return super().__getattribute__(item)

    def _forward(self, X, **kwargs):
        return self.estimator.predict(X, **kwargs)

    def _fit(  # type: ignore[override]
        self,
        X,
        y=None,
        select: t.Optional[Select] = None,
        db: t.Optional[DB] = None,
        configuration: t.Optional[SklearnTrainingConfiguration] = None,
        validation_sets: t.Optional[t.Sequence[str]] = None,
        metrics: t.Optional[t.Sequence[Metric]] = None,
        data_prefetch: bool = False,
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
                if isinstance(y_preprocess, Artifact):
                    y_preprocess = y_preprocess.artifact
            if select is None or db is None:
                raise ValueError('Neither select nor db can be None')
            X, y = get_data_from_query(
                select=select,
                db=db,
                X=X,
                y=y,
                y_preprocess=y_preprocess,
                preprocess=(
                    self.preprocess.artifact  # type: ignore[union-attr]
                    if self.preprocess
                    else lambda x: x
                ),
            )
        if self.training_configuration is not None:
            to_return = self.estimator.fit(
                X=X,
                y=y,
                **self.training_configuration.get(  # type: ignore[union-attr]
                    'fit_params', {}
                ),
            )
        else:
            to_return = self.estimator.fit(X, y)

        if validation_sets is not None:
            results = {}
            for validation_set in validation_sets:
                results.update(
                    self._validate(
                        db=db,
                        validation_set=validation_set,  # type: ignore[arg-type]
                        metrics=metrics,  # type: ignore[arg-type]
                    )
                )
            self.append_metrics(results)
        if db is not None:
            db.replace(self, upsert=True)
        return to_return
