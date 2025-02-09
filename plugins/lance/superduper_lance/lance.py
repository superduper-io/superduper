import os
import typing as t

import lance
import numpy as np
import pyarrow as pa
from superduper.backends.base.vector_search import (
    BaseVectorSearcher,
    VectorIndexMeasureType,
    VectorItem,
)


class LanceVectorSearcher(BaseVectorSearcher):
    """
    Implementation of a vector index using the ``lance`` library.

    :param uuid: Unique string identifier of index
    :param dimensions: Dimension of the vector embeddings in the Lance dataset
    :param measure: measure to assess similarity
    """

    def __init__(
        self,
        uuid: str,
        dimensions: int,
        measure: t.Optional[str] = None,
    ):
        self.dataset_path = os.path.join(
            os.environ.get(
                'SUPERDUPER_LANCE_HOME',
                os.path.expanduser('~/.superduper/vector_indices'),
            ),
            f"{uuid}.lance",
        )
        self.dimensions = dimensions
        self.measure = (
            measure.name if isinstance(measure, VectorIndexMeasureType) else measure
        )

    def initialize(self, db):
        """Initialize the vector index."""
        pass

    @property
    def dataset(self):
        """Return the Lance dataset."""
        if not os.path.exists(self.dataset_path):
            self._create_or_append_to_dataset([], [], mode='create')
        return lance.dataset(self.dataset_path)

    def __len__(self):
        return self.dataset.count_rows()

    def _create_or_append_to_dataset(self, vectors, ids, mode: str = 'upsert'):
        type = pa.list_(
            pa.field('values', pa.float32(), nullable=False), self.dimensions
        )
        vectors = self.to_list(vectors)
        _vecs = pa.array([v for v in vectors], type=type)
        _ids = pa.array(ids, type=pa.string())
        _table = pa.Table.from_arrays([_ids, _vecs], names=['id', 'vector'])

        if mode == 'upsert':
            dataset = lance.dataset(self.dataset_path)
            dataset.merge_insert(
                "id"
            ).when_matched_update_all().when_not_matched_insert_all().execute(_table)
        else:
            lance.write_dataset(_table, self.dataset_path, mode=mode)

    def add(self, items: t.Sequence[VectorItem], cache: bool = False) -> None:
        """Add vectors to the index.

        :param items: List of vectors to add
        :param cache: Cache vectors.
        """
        ids = [item.id for item in items]
        vectors = [item.vector for item in items]
        self._create_or_append_to_dataset(vectors, ids, mode='append')

    def delete(self, ids: t.Sequence[str]) -> None:
        """Delete vectors from the index.

        :param ids: List of IDs to delete
        """
        to_remove = ", ".join(f"'{str(id)}'" for id in ids)
        self.dataset.delete(f"id IN ({to_remove})")

    def find_nearest_from_id(
        self,
        _id,
        n: int = 100,
        within_ids: t.Sequence[str] = (),
    ) -> t.Tuple[t.List[str], t.List[float]]:
        """Find the nearest vectors to a given ID.

        :param _id: ID to search
        :param n: Number of results to return
        :param within_ids: List of IDs to search within
        """
        # The ``lance`` file format has been specifically designed for fast
        # random access. The logic to take advantage of this is implemented
        # by the ``.take`` method.

        # See https://blog.lancedb.com/benchmarking-random-access-in-lance-ed690757a826
        # for further details.

        vector = self.dataset.take([int(_id)], columns=['vector']).to_pydict()[
            'vector'
        ][0]
        return self.find_nearest_from_array(vector, n=n, within_ids=within_ids)

    def find_nearest_from_array(
        self,
        h: np.typing.ArrayLike,
        n: int = 100,
        within_ids: t.Sequence[str] = (),
    ) -> t.Tuple[t.List[str], t.List[float]]:
        """Find the nearest vectors to a given vector.

        :param h: Vector to search
        :param n: Number of results to return
        :param within_ids: List of IDs to search within
        """
        # NOTE: filter is currently applied AFTER vector-search
        # See https://lancedb.github.io/lance/api/python/lance.html#lance.dataset.LanceDataset.scanner
        if within_ids:
            if isinstance(within_ids, (list, set)):
                within_ids = tuple(within_ids)
            assert (
                type(within_ids) is tuple
            ), 'within_ids must be a [tuple | list | set] for lance sql parser'
            result = self.dataset.to_table(
                columns=['id'],
                nearest={
                    'column': 'vector',
                    'q': h,
                    'k': n,
                    'metric': self.measure,
                },
                filter=f"id in {within_ids}",
                prefilter=True,
                offset=0,
            )
        else:
            result = self.dataset.to_table(
                columns=['id'],
                nearest={"column": 'vector', "q": h, "k": n, 'metric': self.measure},
                offset=0,
            )
        ids = result['id'].to_pylist()
        distances = result['_distance'].to_pylist()
        scores = self._convert_distances_to_scores(distances)
        return ids, scores

    def _convert_distances_to_scores(self, distances: list[float]) -> list[float]:
        if self.measure == "cosine":
            scores = [1 - d for d in distances]
        elif self.measure == "l2":
            scores = [1 - d for d in distances]
        elif self.measure == "dot":
            scores = [-d for d in distances]
        else:
            scores = distances

        return scores
