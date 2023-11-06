import typing as t

import numpy

from superduperdb import logging
from superduperdb.vector_search.base import BaseVectorSearcher, VectorItem, measures


class InMemoryVectorSearcher(BaseVectorSearcher):
    """
    Simple hash-set for looking up with vector similarity.

    :param identifier: Unique string identifier of index
    :param h: array/ tensor of vectors
    :param index: list of IDs
    :param measure: measure to assess similarity
    """

    name = 'vanilla'

    def __init__(
        self,
        identifier: str,
        dimensions: int,
        h: t.Optional[numpy.ndarray] = None,
        index: t.Optional[t.List[str]] = None,
        measure: t.Union[str, t.Callable] = 'cosine',
    ):
        self.identifier = identifier
        self.dimensions = dimensions

        if h is not None:
            assert index is not None
            self._setup(h, index)
        else:
            self.h_list = None
            self.h = None
            self.index = None
            self.lookup = None

        self.measure = measure
        if isinstance(measure, str):
            self.measure = measures[measure]

        self.identifier = identifier

    def __len__(self):
        return self.h.shape[0]

    def _setup(self, h, index):
        self.h_list = h if isinstance(h, list) else h.tolist()
        self.h = numpy.array(h) if not isinstance(h, numpy.ndarray) else h
        self.index = index
        self.lookup = dict(zip(index, range(len(index))))

    def find_nearest_from_id(self, _id, n=100):
        return self.find_nearest_from_array(self.h[self.lookup[_id]], n=n)

    def find_nearest_from_array(self, h, n=100, within_ids=None):
        h = self.to_numpy(h)[None, :]
        if within_ids:
            ix = list(map(self.lookup.__getitem__, within_ids))
            similarities = self.measure(h, self.h[ix, :])  # mypy: ignore
        else:
            similarities = self.measure(h, self.h)  # mypy: ignore
        similarities = similarities[0, :]
        logging.debug(similarities)
        scores = -numpy.sort(-similarities)
        ix = numpy.argsort(-similarities)[:n]
        ix = ix.tolist()
        scores = scores.tolist()
        _ids = [self.index[i] for i in ix]
        return _ids, scores

    def add(self, items: t.Sequence[VectorItem]) -> None:
        index = [item.id for item in items]
        h = numpy.stack([item.vector for item in items])

        if self.h is not None:
            old_not_in_new = list(set(self.index) - set(index))
            ix_old = [self.lookup[_id] for _id in old_not_in_new]
            h = numpy.concatenate((self.h[ix_old], h), axis=0)
            index = [self.index[i] for i in ix_old] + index

        return self._setup(h, index)

    def delete(self, ids):
        ix = list(map(self.lookup.__getitem__, ids))
        h = numpy.delete(self.h, ix, axis=0)
        index = [_id for _id in self.index if _id not in set(ids)]
        self._setup(h, index)
