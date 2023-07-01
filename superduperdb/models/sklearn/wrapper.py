import numpy
import typing as t

from sklearn.pipeline import Pipeline as BasePipeline
from sklearn.base import BaseEstimator

from superduperdb.core.model import Model, TrainingConfiguration
from superduperdb.core.metric import Metric
from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.datalayer.base.query import Select
from superduperdb.datalayer.query_dataset import QueryDataset
from tqdm import tqdm


# TODO fix the tests for this one, before moving onto PyTorch pipeline etc.


def postprocess(f):
    f.superduper_postprocess = True
    return f


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


class SklearnTrainingConfiguration(TrainingConfiguration):
    def __init__(
        self,
        identifier,
        fit_params=None,
        predict_params=None,
        y_preprocess=None,
    ):
        super().__init__(
            identifier,
            fit_params=fit_params or {},
            predict_params=predict_params or {},
            y_preprocess=y_preprocess,
        )


class Base(Model):
    def fit(
        self,
        X,
        y=None,
        select: t.Optional[Select] = None,
        database: t.Optional[BaseDatabase] = None,
        training_configuration: t.Optional[SklearnTrainingConfiguration] = None,
        validation_sets: t.Optional[t.List[str]] = None,
        metrics: t.Optional[t.List[Metric]] = None,
    ):
        if training_configuration is not None:
            self.training_configuration = training_configuration
        if select is not None:
            self.training_select = select
        if isinstance(X, str):
            self.training_keys = [X, y] if y is not None else [X]
        if isinstance(X, str):
            y_preprocess = None
            if self.training_configuration is not None:
                y_preprocess = self.training_configuration.get('y_preprocess', None)
            # ruff: noqa: E501
            X, y = get_data_from_query(
                select=select, X=X, y=y, y_preprocess=y_preprocess  # type: ignore[arg-type]
            )
        if self.training_configuration is not None:
            return self.object.fit(
                X, y, **self.training_configuration.get('fit_params', {})
            )
        else:
            return self.object.fit(X, y)


class Estimator(Base):
    def __init__(
        self,
        estimator: BaseEstimator,
        identifier: str,
        training_configuration: t.Optional[SklearnTrainingConfiguration] = None,
        training_select: t.Optional[Select] = None,
        X: t.Optional[str] = None,
        y: t.Optional[str] = None,
    ):
        super().__init__(object=estimator, identifier=identifier)
        self.training_configuration = training_configuration
        self.training_select = training_select
        self.X = X
        self.y = y

    def __getattr__(self, item):
        if item in ['predict', 'transform', 'predict_proba', 'score']:
            return getattr(self.object, item)
        else:
            return super().__getattribute__(item)


class Pipeline(Base):
    def __init__(
        self,
        identifier,
        steps,
        memory=None,
        verbose=False,
        encoder=None,
        training_configuration: t.Optional[SklearnTrainingConfiguration] = None,
        training_select: t.Optional[Select] = None,
        X: t.Optional[str] = None,
        y: t.Optional[str] = None,
    ):
        standard_steps = [
            i
            for i, s in enumerate(steps)
            if not getattr(s[1], 'superduper_postprocess', False)
        ]
        postprocess_steps = [
            i
            for i, s in enumerate(steps)
            if getattr(s[1], 'superduper_postprocess', False)
        ]
        if postprocess_steps:
            msg = 'Postprocess steps must go after preprocess steps'
            assert max(standard_steps) < min(postprocess_steps), msg

        pipeline = BasePipeline(
            [steps[i] for i in standard_steps], memory=memory, verbose=verbose
        )
        self.postprocess_steps = [steps[i] for i in postprocess_steps]

        training_keys = None
        if X and y is None:
            training_keys = [X]
        elif X and y:
            training_keys = [X, y]  # type: ignore[list-item]
        Model.__init__(
            self,
            pipeline,
            identifier,
            encoder=encoder,
            training_configuration=training_configuration,
            training_select=training_select,
            training_keys=training_keys,
        )

    def __getattr__(self, item):
        if item in ['predict_proba', 'score']:
            return getattr(self.object, item)
        else:
            return super().__getattribute__(item)

    def predict(self, X, **predict_params):
        out = self.object.predict(X, **predict_params).tolist()
        if self.postprocess_steps:
            for s in self.postprocess_steps:
                if hasattr(s[1], 'transform'):
                    out = s[1].transform(s)
                elif callable(s[1]):
                    out = s[1](out)
                else:
                    raise Exception('Unexpected postprocessing transform')
        return out
