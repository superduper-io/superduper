import numpy
import typing as t

from sklearn.pipeline import Pipeline as BasePipeline
from sklearn.base import BaseEstimator

from superduperdb.core.model import Model
from superduperdb.core import TrainingConfiguration
from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.datalayer.base.query import Select
from superduperdb.misc import progress
from superduperdb.training.query_dataset import QueryDataset


# TODO fix the tests for this one, before moving onto PyTorch pipeline etc.


def postprocess(f):
    f.superduper_postprocess = True
    return f


def get_data_from_query(select: Select, X: str, y: t.Optional[str] = None,
                        y_preprocess: t.Optional[t.Callable] = None):
    data = QueryDataset(
        select=select,
        keys=[X] if y is None else [X, y],
    )
    documents = []
    for i in progress.progressbar(range(len(data))):
        r = data[i]
        documents.append(r)
    X = [r[X] for r in documents]
    if isinstance(X[0], numpy.ndarray):
        X = numpy.stack(X)
    if y is not None:
        y = [r[y] for r in documents]
        if y_preprocess is not None:
            y = [y_preprocess(item) for item in y]
        if isinstance(y[0], numpy.ndarray):
            y = numpy.stack(y)
    return X, y


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
    ):
        if training_configuration is not None:
            self.training_configuration = training_configuration
        if select is not None:
            self.training_select = select
        if isinstance(X, str):
            self.training_keys = {'X': X, 'y': y} if y is not None else {'X': X}
        if isinstance(X, str):
            y_preprocess = None
            if self.training_configuration is not None:
                y_preprocess = self.training_configuration.get('y_preprocess', None)
            X, y = get_data_from_query(select=select, X=X, y=y, y_preprocess=y_preprocess)
        if self.training_configuration is not None:
            return self.object.fit(X, y, **self.training_configuration.get('fit_params', {}))
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
        Model.__init__(
            self,
            pipeline,
            identifier,
            encoder=encoder,
            training_configuration=training_configuration,
            training_select=training_select,
            training_keys={'X': X, 'y': y}
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
