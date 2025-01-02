import typing as t

import numpy

from superduper import logging
from superduper.backends.base.vector_search import (
    BaseVectorSearcher,
    VectorItem,
    VectorSearchBackend,
    measures,
)

if t.TYPE_CHECKING:
    from superduper import Component


class LocalVectorSearchBackend(VectorSearchBackend):
    """Local vector search backend.

    :param searcher_impl: class to use for requesting similarities
    """

    def __init__(self, searcher_impl: BaseVectorSearcher):
        self._cache = {}
        self.searcher_impl = searcher_impl
        self._identifier_uuid_map: t.Dict = {}
        self._db = None

    def _put(self, vector_index):
        searcher = self.searcher_impl.from_component(vector_index)
        self._cache[vector_index.identifier] = searcher
        self._identifier_uuid_map[vector_index.identifier] = vector_index.uuid

    def initialize(self):
        """Initialize the vector search."""
        for identifier in self.db.show('vector_index'):
            try:
                vector_index = self.db.load('vector_index', identifier=identifier)
                self._put(vector_index)
                vector_index.copy_vectors()
            except FileNotFoundError:
                logging.error(
                    f'Could not load vector index: {identifier} '
                    'Is the artifact store correctly configured?'
                )
                continue
            except TypeError as e:
                import traceback

                logging.error(f'Could not load vector index: {identifier} ' f'{e}')
                logging.error(traceback.format_exc())
                continue

    # TODO needed?
    def __contains__(self, item):
        return item in self._cache

    def list_components(self):
        """List components."""
        return list(self._cache.keys())

    def list_uuids(self):
        """List UUIDs of components."""
        return list(self._identifier_uuid_map.values())

    def __delitem__(self, identifier):
        del self._cache[identifier]
        del self._identifier_uuid_map[identifier]

    def __getitem__(self, identifier):
        return self._cache[identifier]

    def drop(self, component: t.Optional['Component'] = None):
        """Drop the CDC.

        :param component: Component to remove.
        """
        # TODO: drop actual vector search not the cache
        if component is None:
            self._cache = {}
        else:
            del self._cache[component.identifier]
            del self._identifier_uuid_map[component.identifier]


class InMemoryVectorSearcher(BaseVectorSearcher):
    """
    Simple hash-set for looking up with vector similarity.

    :param uuid: Unique string identifier of index
    :param dimensions: Dimension of the vector embeddings
    :param measure: measure to assess similarity
    """

    def __init__(
        self,
        uuid: str,
        dimensions: int,
        measure: str = 'cosine',
    ):
        self.uuid = uuid
        self.dimensions = dimensions

        self._cache: t.Sequence[VectorItem] = []
        self._CACHE_SIZE = 10000

        assert isinstance(measure, str)

        self.measure_name = measure
        self.measure = measures[measure]

        self.h = None
        self.index = None
        self.lookup = None

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

    def initialize(self, vector_index):
        """Initialize the vector index."""
        vector_index.copy_vectors()

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
