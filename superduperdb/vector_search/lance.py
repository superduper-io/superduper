import os
import typing as t

import lance
import numpy
import pyarrow as pa

from superduperdb import CFG
from superduperdb.vector_search.base import BaseVectorSearcher, VectorItem


class LanceVectorSearcher(BaseVectorSearcher):
    """
    Implementation of a vector index using the ``lance`` library.

    :param identifier: Unique string identifier of index
    :param dimensions: Dimension of the vector embeddings in the Lance dataset
    :param h: ``torch.Tensor``
    :param index: list of IDs
    :param measure: measure to assess similarity
    """

    def __init__(
        self,
        identifier: str,
        dimensions: int,
        h: t.Optional[numpy.ndarray] = None,
        index: t.Optional[t.List[str]] = None,
        measure: t.Optional[str] = None,
    ):
        self.dataset_path = os.path.join(CFG.lance_home, f'{identifier}.lance')
        self.dimensions = dimensions
        self._created = False
        self.measure = measure
        if h is not None:
            self._create_or_append_to_dataset(h, index, mode='create')

    @property
    def dataset(self):
        if not os.path.exists(self.dataset_path):
            self._create_or_append_to_dataset([], [])
        return lance.dataset(self.dataset_path)

    def __len__(self):
        return self.dataset.count_rows()

    def _create_or_append_to_dataset(self, vectors, ids, mode: str = 'create'):
        if not self._created:
            if not os.path.exists(self.dataset_path):
                os.makedirs(self.dataset_path, exist_ok=True)
                mode = 'create'
            else:
                self._created = True
                mode = 'append'
        type = pa.list_(
            pa.field('values', pa.float32(), nullable=False), self.dimensions
        )
        vectors = self.to_list(vectors)
        _vecs = pa.array([v for v in vectors], type=type)
        _ids = pa.array(ids, type=pa.string())
        _table = pa.Table.from_arrays([_ids, _vecs], names=['id', 'vector'])
        lance.write_dataset(_table, self.dataset_path, mode=mode)
        self._created = True

    def add(self, items: t.Sequence[VectorItem]) -> None:
        ids = [item.id for item in items]
        vectors = [item.vector for item in items]
        self._create_or_append_to_dataset(vectors, ids, mode='append')

    def delete(self, ids: t.Sequence[str]) -> None:
        to_remove = ", ".join(f"'{str(id)}'" for id in ids)
        self.dataset.delete(f"id IN ({to_remove})")

    def find_nearest_from_id(
        self,
        _id,
        n: int = 100,
        within_ids: t.Sequence[str] = (),
    ) -> t.Tuple[t.List[str], t.List[float]]:
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
        h: numpy.typing.ArrayLike,
        n: int = 100,
        within_ids: t.Sequence[str] = (),
    ) -> t.Tuple[t.List[str], t.List[float]]:
        # NOTE: filter is currently applied AFTER vector-search
        # See https://lancedb.github.io/lance/api/python/lance.html#lance.dataset.LanceDataset.scanner
        if within_ids:
            assert (
                type(within_ids) == tuple
            ), 'within_ids must be a tuple for lance sql parser'
            result = self.dataset.to_table(
                columns=['id'],
                nearest={
                    'column': 'vector',
                    'q': h,
                    'k': n,
                    'metric': self.measure,
                },
                filter=f"id in {within_ids}",
                offset=0,
            )
        else:
            result = self.dataset.to_table(
                columns=['id'],
                nearest={"column": 'vector', "q": h, "k": n},
                offset=0,
            )
        ids = result['id'].to_pylist()
        scores = result['_distance'].to_pylist()
        return ids, scores
