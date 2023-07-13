import typing as t
from collections import defaultdict

import numpy
from superduperdb.core.metric import Metric
from superduperdb.core.model import Model, ModelEnsemble
from superduperdb.datalayer.query_dataset import QueryDataset
from superduperdb.vector_search.base import BaseVectorIndex, VectorCollection

from superduperdb.vector_search import VanillaVectorIndex


class PatK:
    def __init__(self, k):
        self.k = k

    def __call__(self, x, y):
        return y in x[: self.k]


class VectorSearchPerformance:
    def __init__(
        self,
        index_key: str,
        measure: t.Union[str, t.Callable],
        splitter: t.Optional[t.Callable] = None,
        hash_set_cls: BaseVectorIndex = VanillaVectorIndex,
        predict_kwargs: t.Optional[t.Dict] = None,
        compatible_keys: t.Optional[t.List] = None,
        vector_collection: t.Optional[VectorCollection] = None,
    ):
        self.measure = measure
        self.hash_set_cls = hash_set_cls
        self.splitter = splitter
        self.predict_kwargs = predict_kwargs
        self.index_key = index_key
        self.compatible_keys = compatible_keys
        self.vector_collection = vector_collection

    def __call__(
        self,
        validation_data: t.Union[QueryDataset, t.List],
        model: t.Union[Model, ModelEnsemble],
        metrics: t.List[Metric],
    ) -> t.Dict[str, t.List]:
        if hasattr(model, 'train_X') and model.train_X is not None:
            keys = model.train_X
        else:
            keys = (
                self.index_key,
                *(self.compatible_keys if self.compatible_keys is not None else ()),
            )
        if isinstance(model, ModelEnsemble):
            msg = 'Model ensemble should only be used in case of multi-model retrieval'
            assert len(model.models) > 1, msg
            models = model.models
        else:
            models = [model for _ in range(2)]
            keys = [keys[0] for _ in range(2)]

        if isinstance(model, ModelEnsemble):
            ix_index = next(i for i, k in enumerate(keys) if k == self.index_key)
            ix_compatible = next(i for i, k in enumerate(keys) if k != self.index_key)
            single_model = False
        else:
            msg = 'Single model retrieval must be tested using a "splitter"'
            assert self.splitter is not None, msg
            ix_index = 0
            ix_compatible = 0
            single_model = True

        inputs: t.List[t.List] = [[] for _ in models]
        for i in range(len(validation_data)):
            r = validation_data[i]
            if single_model:
                all_r = self.splitter(r)
            else:
                all_r = [r for _ in models]

            for j, m in enumerate(models):
                inputs[j].append(all_r[j][keys[j]])

        random_order = numpy.random.permutation(len(inputs[0]))
        inputs = [[x[i] for i in random_order] for x in inputs]
        if self.vector_collection is None:
            predictions = [
                model.predict(inputs[i], **(self.predict_kwargs or {}))
                for i, model in enumerate(models)
            ]
            vi = self.hash_set_cls(  # type: ignore[operator]
                predictions[ix_index],
                list(range(len(predictions[0]))),
                self.measure,
            )
        else:
            raise NotImplementedError  # TODO
        metric_values = defaultdict(lambda: [])
        for i in range(len(predictions[ix_compatible])):
            ix, _ = vi.find_nearest_from_array(predictions[ix_compatible][i], n=100)
            for metric in metrics:
                metric_values[metric.identifier].append(metric(ix, i))
        for k in metric_values:
            metric_values[k] = sum(metric_values[k]) / len(metric_values[k])

        return metric_values


def validate_vector_search(
    validation_data,
    models,
    keys,
    metrics,
    hash_set_cls,
    measure,
    splitter=None,
    predict_kwargs=None,
):
    inputs = [[] for _ in models]
    for i in range(len(validation_data)):
        r = validation_data[i]
        if splitter is not None:
            all_r = splitter(r)
        else:
            all_r = [r for _ in models]
        for j, m in enumerate(models):
            inputs[j].append(all_r[j][keys[j]])

    random_order = numpy.random.permutation(len(inputs[0]))
    inputs = [[x[i] for i in random_order] for x in inputs]
    predictions = [
        model.predict(inputs[i], **(predict_kwargs or {}))
        for i, model in enumerate(models)
    ]
    h = hash_set_cls(predictions[0], list(range(len(predictions[0]))), measure)
    metric_values = defaultdict(lambda: [])
    for i in range(len(predictions[0])):
        ix, _ = h.find_nearest_from_array(predictions[0][i], n=100)
        for metric in metrics:
            metric_values[metric.identifier].append(metric(ix, i))

    for k in metric_values:
        metric_values[k] = sum(metric_values[k]) / len(metric_values[k])

    return metric_values
