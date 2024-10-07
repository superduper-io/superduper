import typing as t

import click
import numpy

from superduper import logging
from superduper.backends.base.cluster import Cluster
from superduper.backends.base.vector_search import (
    BaseVectorSearcher,
    VectorItem,
    measures,
)
from superduper.backends.local.cache import LocalCache
from superduper.backends.local.cdc import LocalCDCBackend
from superduper.backends.local.compute import LocalComputeBackend
from superduper.backends.local.crontab import LocalCrontabBackend
from superduper.backends.local.queue import LocalQueuePublisher
from superduper.backends.local.vector_search import LocalVectorSearchBackend
from superduper.misc.plugins import load_plugin


class LocalCluster(Cluster):
    """Local cluster for running infra locally.

    :param compute: The compute backend.
    :param cache: The cache backend.
    :param queue: The queue backend.
    :param vector_search: The vector search backend.
    :param cdc: The change data capture backend.
    :param crontab: The crontab backend.
    """

    @classmethod
    def build(cls, CFG, **kwargs):
        """Build the local cluster."""
        searcher_impl = load_plugin(CFG.vector_search_engine).VectorSearcher
        return LocalCluster(
            compute=LocalComputeBackend(),
            cache=LocalCache(),
            queue=LocalQueuePublisher(),
            vector_search=LocalVectorSearchBackend(searcher_impl=searcher_impl),
            cdc=LocalCDCBackend(),
            crontab=LocalCrontabBackend(),
        )

    def drop(self, force: bool = False):
        """Drop the cluster."""
        if not force:
            if not click.confirm(
                "Are you sure you want to drop the cache? ",
                default=False,
            ):
                logging.warn("Aborting...")
        return self.cache.drop()


class InMemoryVectorSearcher(BaseVectorSearcher):
    """
    Simple hash-set for looking up with vector similarity.

    :param identifier: Unique string identifier of index
    :param dimensions: Dimension of the vector embeddings
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
        measure: str = 'cosine',
    ):
        self.identifier = identifier
        self.dimensions = dimensions
        self._cache: t.Sequence[VectorItem] = []
        self._CACHE_SIZE = 10000

        assert isinstance(measure, str)

        self.measure_name = measure
        self.measure = measures[measure]

        if h is not None:
            assert index is not None
            self._setup(h, index)
        else:
            self.h = None
            self.index = None
            self.lookup = None

        self.identifier = identifier

    def __len__(self):
        if self.h is not None:
            return self.h.shape[0]
        else:
            return 0

    def _setup(self, h, index):
        h = numpy.array(h) if not isinstance(h, numpy.ndarray) else h

        if self.measure_name == 'cosine':
            # Normalization is required for cosine, hence preparing
            # all vectors in advance.
            h = h / numpy.linalg.norm(h, axis=1)[:, None]
        self.h = h
        self.index = index
        self.lookup = dict(zip(index, range(len(index))))

    def find_nearest_from_id(self, _id, n=100):
        """Find the nearest vectors to the given ID.

        :param _id: ID of the vector
        :param n: number of nearest vectors to return
        """
        self.post_create()
        return self.find_nearest_from_array(self.h[self.lookup[_id]], n=n)

    def find_nearest_from_array(self, h, n=100, within_ids=None):
        """Find the nearest vectors to the given vector.

        :param h: vector
        :param n: number of nearest vectors to return
        :param within_ids: list of IDs to search within
        """
        self.post_create()

        if self.h is None:
            logging.error(
                'Tried to search on an empty vector database',
                'Vectors are not yet loaded in vector database.',
                '\nPlease check if model outputs are ready.',
            )
            return [], []

        h = self.to_numpy(h)[None, :]
        if within_ids:
            ix = list(map(self.lookup.__getitem__, within_ids))
            similarities = self.measure(h, self.h[ix, :])  # mypy: ignore
        else:
            similarities = self.measure(h, self.h)  # mypy: ignore
        similarities = similarities[0, :]
        logging.debug(similarities)
        scores = -numpy.sort(-similarities)
        ## different ways of handling
        if within_ids:
            top_n_idxs = numpy.argsort(-similarities)[:n]
            ix = [ix[i] for i in top_n_idxs]
        else:
            ix = numpy.argsort(-similarities)[:n]
            ix = ix.tolist()
        scores = scores.tolist()
        _ids = [self.index[i] for i in ix]
        return _ids, scores

    def add(self, items: t.Sequence[VectorItem] = (), cache: bool = False) -> None:
        """Add vectors to the index.

        Only adds to cache if cache is not full.

        :param items: List of vectors to add
        :param cache: Flush the cache and add all vectors
        """
        if not cache:
            return self._add(items)

        for item in items:
            self._cache.append(item)
        if len(self._cache) == self._CACHE_SIZE:
            self._add(self._cache)
            self._cache = []

    def post_create(self):
        """Post create method to incorporate remaining vectors to be added in cache."""
        if self._cache:
            self._add(self._cache)
            self._cache = []

    def _add(self, items: t.Sequence[VectorItem]) -> None:
        index = [item.id for item in items]
        h = numpy.stack([item.vector for item in items])

        if self.h is not None:
            old_not_in_new = list(set(self.index) - set(index))
            ix_old = [self.lookup[_id] for _id in old_not_in_new]
            h = numpy.concatenate((self.h[ix_old], h), axis=0)
            index = [self.index[i] for i in ix_old] + index

        return self._setup(h, index)

    def delete(self, ids):
        """Delete vectors from the index.

        :param ids: List of IDs to delete
        """
        self.post_create()
        ix = list(map(self.lookup.__getitem__, ids))
        h = numpy.delete(self.h, ix, axis=0)
        index = [_id for _id in self.index if _id not in set(ids)]
        self._setup(h, index)
