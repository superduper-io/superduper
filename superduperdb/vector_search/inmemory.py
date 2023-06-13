from contextlib import contextmanager
from typing import List, Sequence, Iterator, Dict, Union

import numpy
from readerwriterlock import rwlock

from ..misc.config import VectorSearchConfig
from .vanilla.hashes import VanillaHashSet
from .base import (
    VectorIndex,
    VectorIndexResult,
    VectorIndexId,
    VectorIndexManager,
    VectorIndexItem,
    VectorIndexItemId,
    VectorIndexItemNotFound,
    VectorIndexMeasureFunction,
)


class InMemoryVectorIndex(VectorIndex):
    def __init__(
        self, *, dimensions: int, measure: Union[str, VectorIndexMeasureFunction] = 'l2'
    ) -> None:
        super().__init__()
        self._index = VanillaHashSet(
            numpy.empty((0, dimensions), dtype='float32'),
            [],
            measure,
        )
        self._lock = rwlock.RWLockFair()

    @contextmanager
    def init(self) -> Iterator["VectorIndex"]:
        yield self

    def add(self, items: Sequence[VectorIndexItem]) -> None:
        for item in items:
            with self._lock.gen_wlock():
                self._add(item)

    def _add(self, item: VectorIndexItem) -> None:
        ix = self._index.lookup.get(item.id)
        if ix is not None:
            self._index.h[ix] = item.vector
        else:
            self._index.index.append(item.id)
            self._index.lookup[item.id] = len(self._index.lookup)
            self._index.h = numpy.append(self._index.h, [item.vector], axis=0)

    def find_nearest_from_id(
        self, identifier: VectorIndexItemId, *, limit: int = 100, offset: int = 0
    ) -> List[VectorIndexResult]:
        with self._lock.gen_rlock():
            try:
                ids, scores = self._index.find_nearest_from_id(
                    identifier, n=limit + offset
                )
            except KeyError:
                raise VectorIndexItemNotFound()
            return self._convert_ids_scores_to_results(ids[offset:], scores[offset:])

    def find_nearest_from_array(
        self, array: numpy.ndarray, *, limit: int = 100, offset: int = 0
    ) -> List[VectorIndexResult]:
        with self._lock.gen_rlock():
            ids, scores = self._index.find_nearest_from_hash(array, n=limit + offset)
            return self._convert_ids_scores_to_results(ids[offset:], scores[offset:])

    def _convert_ids_scores_to_results(
        self, ids: numpy.ndarray, scores: numpy.ndarray
    ) -> List[VectorIndexResult]:
        results: List[VectorIndexResult] = []
        for ix, score in zip(ids, scores):
            results.append(
                VectorIndexResult(
                    id=self._index.lookup[ix],
                    score=score,
                )
            )
        return results


class InMemoryVectorIndexManager(VectorIndexManager):
    def __init__(self, *, config: VectorSearchConfig) -> None:
        self._config = config
        self._indices: Dict[VectorIndexId, VectorIndex] = {}

    @contextmanager
    def init(self) -> Iterator["VectorIndexManager"]:
        yield self

    @contextmanager
    def get_index(
        self, identifier: VectorIndexId, *, dimensions: int
    ) -> Iterator[VectorIndex]:
        index = self._indices.get(identifier)
        if not index:
            index = self._indices[identifier] = InMemoryVectorIndex(
                dimensions=dimensions
            )
        yield index
