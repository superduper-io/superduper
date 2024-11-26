import dataclasses as dc
import typing as t

import numpy
from sklearn.base import BaseEstimator
from superduper import logging
from superduper.backends.query_dataset import QueryDataset
from superduper.base.datalayer import Datalayer
from superduper.components.datatype import pickle_serializer
from superduper.components.model import (
    Model,
    ModelInputType,
    Signature,
    Trainer,
)
from tqdm import tqdm


class SklearnTrainer(Trainer):
    """A trainer for `sklearn` models.

    :param fit_params: The parameters to pass to `fit`.
    :param predict_params: The parameters to pass to `predict
    :param y_preprocess: The preprocessing function to use for the target.
    """

    fit_params: t.Dict = dc.field(default_factory=dict)
    predict_params: t.Dict = dc.field(default_factory=dict)
    y_preprocess: t.Optional[t.Callable] = None

    @staticmethod
    def _get_data_from_dataset(dataset, X: ModelInputType):
        # X is the same input type as with `db.predict_in_db` etc.
        # However in `sklearn` there are only 2 possibilities
        # ('x', 'y') and 'x'
        # That is why we assert the following
        assert isinstance(X, (tuple, list, str))
        if isinstance(X, (tuple, list)):
            # ('x', 'y')
            assert len(X) == 2
        # else 'x' (str)

        rows = []
        logging.info('Loading dataset into memory for Estimator.fit')
        for i in tqdm(range(len(dataset))):
            rows.append(dataset[i])

        if isinstance(X, str):
            X_arr = rows
        else:
            X_arr = [x[0] for x in rows]
            y_arr = [x[1] for x in rows]

        if isinstance(X_arr[0], numpy.ndarray):
            X_arr = numpy.stack(X_arr)
        if isinstance(X, (tuple, list)):
            y_arr = [r[1] for r in rows]
            if isinstance(y_arr[0], numpy.ndarray):
                y_arr = numpy.stack(y_arr)
            return X_arr, y_arr
        return X_arr, None

    def fit(
        self,
        model: 'Estimator',
        db: Datalayer,
        train_dataset: QueryDataset,
        valid_dataset: QueryDataset,
    ):
        """Fit the model.

        :param model: Model
        :param db: Datalayer
        :param train_dataset: Training dataset
        :param valid_dataset: Validation dataset
        """
        train_X, train_y = self._get_data_from_dataset(
            dataset=train_dataset, X=self.key
        )
        if train_y:
            model.object.fit(train_X, train_y, **self.fit_params)
        else:
            model.object.fit(train_X, **self.fit_params)

        metrics = {}
        if (validation := model.validation) is not None:
            key = validation.key or self.key
            for dataset in validation.datasets:
                dataset_metrics = model.validate(key, dataset, validation.metrics)
                dataset_metrics = {
                    f'{dataset.identifier}/{k}': v for k, v in dataset_metrics.items()
                }
                metrics.update(dataset_metrics)

        model.metric_values = metrics
        db.replace(model)


class Estimator(Model):
    """Estimator model.

    This is a model that can be trained and used for prediction.

    :param object: The estimator object from `sklearn`.
    :param trainer: The trainer to use.
    :param preprocess: The preprocessing function to use.
    :param postprocess: The postprocessing function to use.

    Example:
    -------
    >>> from superduper_sklearn import Estimator
    >>> from sklearn.svm import SVC
    >>> model = Estimator(
    >>>     identifier='test',
    >>>     object=SVC(),
    >>> )

    """

    _fields = {'object': pickle_serializer}

    object: BaseEstimator
    trainer: t.Optional[SklearnTrainer] = None
    preprocess: t.Optional[t.Callable] = None
    postprocess: t.Optional[t.Callable] = None
    signature: Signature = 'singleton'

    def predict(self, X):
        """Predict on a single input.

        :param X: The input to predict on.
        """
        if isinstance(X, numpy.ndarray):
            X = X[None, :]
        if self.preprocess is not None:
            X = self.preprocess(X)
        X = self.object.predict(X, **self.predict_kwargs)[0]
        if self.postprocess is not None:
            X = self.postprocess(X)
        return X

    def predict_batches(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        """Predict on a dataset.

        :param dataset: The dataset to predict on.
        """
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
