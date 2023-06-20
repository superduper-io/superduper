from contextlib import contextmanager
import typing as t

import numpy
from readerwriterlock import rwlock

from ..misc.config import VectorSearchConfig
from .vanilla.hashes import VanillaHashSet
from .base import (
    ArrayLike,
    to_numpy,
    VectorCollection,
    VectorCollectionResult,
    VectorCollectionId,
    VectorDatabase,
    VectorCollectionItem,
    VectorCollectionItemId,
    VectorCollectionItemNotFound,
    VectorIndexMeasure,
    VectorCollectionConfig,
)


class InMemoryVectorCollection(VectorCollection):
    def __init__(self, *, dimensions: int, measure: VectorIndexMeasure = 'l2') -> None:
        super().__init__()
        self._index = VanillaHashSet(
            numpy.empty((0, dimensions), dtype='float32'),
            [],
            measure,
        )
        self._lock = rwlock.RWLockFair()

    @contextmanager
    def init(self) -> t.Iterator["VectorCollection"]:
        yield self

    def add(self, items: t.Sequence[VectorCollectionItem]) -> None:
        for item in items:
            with self._lock.gen_wlock():
                self._add(item)

    def _add(self, item: VectorCollectionItem) -> None:
        ix = self._index.lookup.get(item.id)
        if ix is not None:
            self._index.h[ix] = item.vector  # type: ignore
        else:
            self._index.index.append(item.id)
            self._index.lookup[item.id] = len(self._index.lookup)
            self._index.h = numpy.append(self._index.h, [item.vector], axis=0)

    def find_nearest_from_id(
        self,
        identifier: VectorCollectionItemId,
        *,
        within_ids: t.Sequence[VectorCollectionItemId] = (),
        limit: int = 100,
        offset: int = 0,
    ) -> t.List[VectorCollectionResult]:
        if within_ids:
            raise NotImplementedError("within_ids not supported")
        with self._lock.gen_rlock():
            try:
                ids, scores = self._index.find_nearest_from_id(
                    identifier, n=limit + offset
                )
            except KeyError:
                raise VectorCollectionItemNotFound()
            return self._convert_ids_scores_to_results(ids[offset:], scores[offset:])

    def find_nearest_from_array(
        self,
        array: ArrayLike,
        *,
        within_ids: t.Sequence[VectorCollectionItemId] = (),
        limit: int = 100,
        offset: int = 0,
    ) -> t.List[VectorCollectionResult]:
        if within_ids:
            raise NotImplementedError("within_ids not supported")
        with self._lock.gen_rlock():
            ids, scores = self._index.find_nearest_from_hash(
                to_numpy(array), n=limit + offset
            )
            return self._convert_ids_scores_to_results(ids[offset:], scores[offset:])

    def _convert_ids_scores_to_results(
        self, ids: numpy.ndarray, scores: numpy.ndarray
    ) -> t.List[VectorCollectionResult]:
        results: t.List[VectorCollectionResult] = []
        for id, score in zip(ids, scores):
            results.append(VectorCollectionResult(id=id, score=score))
        return results


class InMemoryVectorDatabase(VectorDatabase):
    def __init__(self, *, config: VectorSearchConfig) -> None:
        self._config = config
        self._collections: t.Dict[VectorCollectionId, VectorCollection] = {}

    @contextmanager
    def init(self) -> t.Iterator["VectorDatabase"]:
        yield self

    @contextmanager
    def get_collection(
        self, config: VectorCollectionConfig
    ) -> t.Iterator[VectorCollection]:
        collection = self._collections.get(config.id)
        if not collection:
            collection = self._collections[config.id] = InMemoryVectorCollection(
                dimensions=config.dimensions, measure=config.measure
            )
        yield collection
