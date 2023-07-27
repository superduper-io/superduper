import typing as t
from collections import defaultdict

import numpy
from superduperdb.core.metric import Metric
from superduperdb.core.model import Model
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
        hash_set_cls: t.Type[BaseVectorIndex] = VanillaVectorIndex,
        predict_kwargs: t.Optional[t.Dict] = None,
        compatible_keys: t.Optional[t.Sequence] = None,
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
        validation_data: t.Union[QueryDataset, t.Sequence],
        model: Model,
        metrics: t.Sequence[Metric],
    ) -> t.Dict[str, t.List]:
        if self.vector_collection is not None:
            raise NotImplementedError  # TODO

        if hasattr(model, 'train_X') and model.train_X is not None:
            keys = model.train_X
        else:
            keys = (
                self.index_key,
                *(self.compatible_keys if self.compatible_keys is not None else ()),
            )
        models = model.models

        ix_index = next(i for i, k in enumerate(keys) if k == self.index_key)
        ix_compatible = next(i for i, k in enumerate(keys) if k != self.index_key)
        single_model = False

        inputs: t.List[t.List] = [[] for _ in models]
        for i in range(len(validation_data)):
            r = validation_data[i]
            if single_model:
                all_r = self.splitter and self.splitter(r)
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

        items = metric_values.items()
        return {k: sum(v) / len(v) for k, v in items}  # type: ignore[misc]
