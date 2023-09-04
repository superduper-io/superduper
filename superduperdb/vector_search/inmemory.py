import typing as t
from contextlib import contextmanager

import numpy
from readerwriterlock import rwlock

from .base import (
    VectorCollection,
    VectorCollectionConfig,
    VectorCollectionId,
    VectorCollectionItem,
    VectorCollectionItemId,
    VectorCollectionItemNotFound,
    VectorCollectionResult,
    VectorDatabase,
    VectorIndexMeasure,
    VectorIndexMeasureType,
    to_numpy,
)
from .table_scan import VanillaVectorIndex


class InMemoryVectorCollection(VectorCollection):
    """
    An in-memory vector collection.

    :param dimensions: The number of dimensions of the vectors in the collection.
    :param measure: The distance measure to use for vector search.
    """

    def __init__(
        self,
        *,
        dimensions: int,
        measure: VectorIndexMeasure = VectorIndexMeasureType.l2,
    ) -> None:
        super().__init__()
        self._index = VanillaVectorIndex(
            numpy.empty((0, dimensions), dtype='float32'),
            [],
            measure,
        )
        self._lock = rwlock.RWLockFair()

    @contextmanager
    def init(self) -> t.Iterator["VectorCollection"]:
        yield self

    def add(self, items: t.Sequence[VectorCollectionItem], **kwargs) -> None:
        """
        Add items to the collection.
        """
        for item in items:
            with self._lock.gen_wlock():
                self._add(item)

    def _add(self, item: VectorCollectionItem) -> None:
        ix = self._index.lookup.get(item.id)
        if ix is not None:
            self._index.h[ix] = item.vector
        else:
            self._index.index.append(item.id)
            self._index.lookup[item.id] = len(self._index.lookup)
            self._index.h_list.append(list(item.vector))

    def find_nearest_from_id(
        self,
        identifier: VectorCollectionItemId,
        *,
        within_ids: t.Sequence[VectorCollectionItemId] = (),
        limit: int = 100,
        offset: int = 0,
    ) -> t.List[VectorCollectionResult]:
        """
        Find items that are nearest to the item with the given identifier.

        :param identifier: identifier of the item
        :param within_ids: identifiers to search within
        :param limit: maximum number of nearest items to return
        :param offset: offset of the first item to return
        """
        with self._lock.gen_rlock():
            try:
                index = self._index if not within_ids else self._index[within_ids]
                ids, scores = index.find_nearest_from_id(identifier, n=limit + offset)
            except KeyError:
                raise VectorCollectionItemNotFound()
            return self._convert_ids_scores_to_results(ids[offset:], scores[offset:])

    def find_nearest_from_array(
        self,
        array: numpy.typing.ArrayLike,
        *,
        within_ids: t.Sequence[VectorCollectionItemId] = (),
        limit: int = 100,
        offset: int = 0,
    ) -> t.List[VectorCollectionResult]:
        """
        Find items that are nearest to the given vector.

        :param array: array representing the vector
        :param within_ids: identifiers to search within
        :param limit: maximum number of nearest items to return
        :param offset: offset of the first item to return
        """

        with self._lock.gen_rlock():
            index = self._index if not within_ids else self._index[within_ids]
            ids, scores = index.find_nearest_from_array(
                to_numpy(array),
                n=limit + offset,
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
    """
    An in-memory vector database.
    """

    def __init__(self, uri: str = 'inmemory://') -> None:
        assert uri.startswith('inmemory://')
        self._collections: t.Dict[VectorCollectionId, VectorCollection] = {}

    def get_table(self, config: VectorCollectionConfig, **kwargs) -> VectorCollection:
        collection = self._collections.get(config.id)
        if not collection:
            collection = self._collections[config.id] = InMemoryVectorCollection(
                dimensions=config.dimensions, measure=config.measure
            )
        return collection
