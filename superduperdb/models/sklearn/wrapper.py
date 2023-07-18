import numpy
import typing as t

from pydantic import Field
from tqdm import tqdm

from superduperdb.core.artifact import Artifact
from superduperdb.core.model import Model, _TrainingConfiguration
from superduperdb.core.metric import Metric
from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.datalayer.base.query import Select
from superduperdb.datalayer.query_dataset import QueryDataset
import dataclasses as dc


def get_data_from_query(
    select: Select,
    X: str,
    y: t.Optional[str] = None,
    y_preprocess: t.Optional[t.Callable] = None,
):
    data = QueryDataset(
        select=select,
        keys=[X] if y is None else [X, y],
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
    fit_params: t.Dict = Field(default_factory=dict)
    predict_params: t.Dict = Field(default_factory=dict)
    y_preprocess: t.Optional[Artifact] = None


@dc.dataclass
class Estimator(Model):
    postprocess: t.Union[Artifact, t.Callable, None] = None

    def __post_init__(self):
        super().__post_init__()
        if self.postprocess and not isinstance(self.postprocess, Artifact):
            self.postprocess = Artifact(artifact=self.postprocess)

    @property
    def estimator(self):
        return self.object.artifact

    def __getattr__(self, item):
        if item in ['transform', 'predict_proba', 'score']:
            return getattr(self.estimator, item)
        else:
            return super().__getattribute__(item)

    def _predict(self, X, **predict_params):
        out = self.estimator.predict(X, **predict_params).tolist()
        if self.postprocess:
            out = self.postprocess(out)
        return out

    def _fit(  # type: ignore[override]
        self,
        X,
        y=None,
        select: t.Optional[Select] = None,
        db: t.Optional[BaseDatabase] = None,
        configuration: t.Optional[SklearnTrainingConfiguration] = None,
        validation_sets: t.Optional[t.List[str]] = None,
        metrics: t.Optional[t.List[Metric]] = None,
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
            y_preprocess = None
            if self.training_configuration is not None:
                y_preprocess = self.training_configuration.get('y_preprocess', None)
                if isinstance(y_preprocess, Artifact):
                    y_preprocess = y_preprocess.artifact
            # ruff: noqa: E501
            X, y = get_data_from_query(
                select=select, X=X, y=y, y_preprocess=y_preprocess  # type: ignore[arg-type]
            )
        if self.training_configuration is not None:
            return self.estimator.fit(
                X=X,
                y=y,
                **self.training_configuration.get('fit_params', {}),
            )
        else:
            return self.estimator.fit(X, y)
