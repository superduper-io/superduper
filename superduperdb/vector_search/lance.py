import os
import typing as t

import lance
import numpy as np
import pyarrow as pa

from superduperdb.vector_search.base import (
    VectorCollection,
    VectorCollectionConfig,
    VectorCollectionItem,
    VectorCollectionItemId,
    VectorCollectionResult,
    VectorDatabase,
)

# TODO: Implement a vector index manager for ``lance`` after tidying
# base classes and interfaces in ``superduperdb.vector_search.base``.


class LanceVectorDatabase(VectorDatabase):
    """A class for managing multiple Lance datasets."""

    def __init__(self, root_dir: str) -> None:
        """
        :param root_dir: root directory that contains all Lance datasets
        """
        assert root_dir.startswith('lance://')
        self.root_dir = root_dir.split('lance://')[1]

    # TODO: ``create`` kwarg is not included in the ABC interface but it is assumed part
    # of the interface in ``superduperdb/container/vector_index.py``! Short-term fix
    # is to include it here but long-term fix is to tidy up the ABC interface.
    def get_table(
        self, config: VectorCollectionConfig, create: bool = False
    ) -> VectorCollection:
        """Retrieve a managed Lance dataset according to the given config."""
        uri = os.path.join(self.root_dir, config.id)
        return LanceVectorCollection(
            uri=uri, dimensions=config.dimensions, measure=config.measure
        )


class LanceVectorCollection(VectorCollection):
    """A class for managing a lance dataset."""

    def __init__(self, uri: str, dimensions: int, measure: str = 'cosine') -> None:
        """
        :param uri: URI of the Lance dataset
        :param dimensions: dimension of the vector embeddings in the Lance dataset
        :param measure: measure which defines strategy to use for vector search.
        """
        self.uri = uri
        self.dimensions = dimensions
        self.measure = measure

    @property
    def _dataset(self) -> lance.LanceDataset:
        try:
            return lance.dataset(self.uri)
        except Exception as e:
            raise Exception(f'Could not load Lance dataset at {self.uri}') from e

    # TODO: ``upsert`` kwarg is not included in the ABC interface but it is assumed part
    # of the interface in ``superduperdb/db/mongodb/cdc/vector_task_factory.py``!
    # Short-term fix is to include it here, long-term fix is to tidy up ABC interface.

    def size(self) -> int:
        """
        Get the number of rows in lance dataset.
        """
        return self._dataset.count_rows()

    def delete_from_ids(self, ids: t.Sequence[str]) -> None:
        """
        Delete vectors from the lance dataset.
        :param ids: t.Sequence of identifiers.
        """
        to_remove = ", ".join(f"'{str(id)}'" for id in ids)
        self._dataset.delete(f"id IN ({to_remove})")

    def add(self, data: t.Sequence[VectorCollectionItem], upsert: bool = False) -> None:
        """
        Add vectors to existing Lance dataset or create a new one if it doesn't exist.

        :param data: t.Sequence of ``VectorCollectionItem`` objects.
        """

        # TODO: Handle floating point bits
        typ = pa.list_(
            pa.field('values', pa.float32(), nullable=False), self.dimensions
        )

        vectors = []
        for d in data:
            vector = d.vector
            if hasattr(vector, 'numpy'):
                vector = vector.numpy()
            vectors.append(vector)

        _vecs = pa.array([vector for vector in vectors], type=typ)
        _ids = pa.array([item.id for item in data], type=pa.string())
        _table = pa.Table.from_arrays([_ids, _vecs], names=['id', 'vector'])

        if not os.path.exists(self.uri):
            if upsert:
                lance.write_dataset(_table, self.uri, mode='create')
            else:
                raise FileNotFoundError('URI is either invalid or does not exists')
        else:
            lance.write_dataset(_table, self.uri, mode='append')

        # TODO: Add logic to optimise Lance dataset for vector search at
        # regular intervals or after a certain number of items have been added.
        # These optimisations include fragment merging, compacting, deleting, etc.
        # See ``lance.DatasetOptimizer`` for further details.

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

        The ``lance`` file format has been specifically designed for fast
        random access. The logic to take advantage of this is implemented
        by the ``.take`` method.

        See https://blog.lancedb.com/benchmarking-random-access-in-lance-ed690757a826
        for further details.

        :param identifier: identifier of the item
        :param within_ids: identifiers to search within
        :param limit: maximum number of nearest items to return
        :param offset: offset of the first item to return
        """

        # ``.take`` expect a list of integers or pyarrow array as input
        vector = self._dataset.take([int(identifier)], columns=['vector']).to_pydict()[
            'vector'
        ][0]
        return self.find_nearest_from_array(
            vector, limit=limit, offset=offset, within_ids=within_ids
        )

    def find_nearest_from_array(
        self,
        array: np.typing.ArrayLike,
        *,
        within_ids: t.Sequence[VectorCollectionItemId] = (),
        limit: int = 100,
        offset: int = 0,
        measure: t.Optional[str] = None,
    ) -> t.List[VectorCollectionResult]:
        """
        Find items that are nearest to the given vector.

        :param array: array representing the vector
        :param within_ids: identifiers to search within
        :param limit: maximum number of nearest items to return
        :param offset: offset of the first item to return
        """

        # NOTE: filter is currently applied AFTER vector-search
        # See https://lancedb.github.io/lance/api/python/lance.html#lance.dataset.LanceDataset.scanner
        if within_ids:
            assert (
                type(within_ids) == tuple
            ), 'within_ids must be a tuple for lance sql parser'
            result = self._dataset.to_table(
                columns=['id'],
                nearest={
                    'column': 'vector',
                    'q': array,
                    'k': limit,
                    'metric': measure if measure else self.measure,
                },
                filter=f"id in {within_ids}",
                offset=offset,
            )
        else:
            result = self._dataset.to_table(
                columns=['id'],
                nearest={"column": 'vector', "q": array, "k": limit},
                offset=offset,
            )

        ids = result['id'].to_pylist()
        scores = result['_distance'].to_pylist()

        return [
            VectorCollectionResult(id=id_, score=(-1 * distance))
            for id_, distance in zip(ids, scores)
        ]
