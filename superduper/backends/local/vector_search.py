import typing as t

import numpy

from superduper import logging
from superduper.backends.base.vector_search import (
    BaseVectorSearcher,
    VectorItem,
    VectorSearchBackend,
    measures,
)
from superduper.base import exceptions

if t.TYPE_CHECKING:
    from superduper import VectorIndex


class LocalVectorSearchBackend(VectorSearchBackend):
    """Local vector search backend.

    :param searcher_impl: class to use for requesting similarities
    """

    def __init__(self, searcher_impl: BaseVectorSearcher):
        super().__init__()
        self.searcher_impl = searcher_impl
        self._db = None

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, value):
        self._db = value
        for tool in self.tools:
            tool.db = value

    def build_tool(self, component):
        searcher = self.searcher_impl.from_component(component)
        return searcher

    def initialize(self):
        """Initialize the vector search."""
        components = []
        from superduper import VectorIndex

        for cls in self.db.show('Table'):
            t = self.db.load('Table', identifier=cls)
            if t.is_component and t.cls is not None:
                if issubclass(t.cls, VectorIndex):
                    components.append(t.identifier)
        for component in components:
            try:
                for identifier in self.db.show(component):
                    try:
                        vector_index = self.db.load(component, identifier=identifier)
                        self.put_component(vector_index)
                        vectors = vector_index.get_vectors()
                        vectors = [VectorItem(**vector) for vector in vectors]
                        self.get_tool(vector_index.uuid).add(vectors)

                    except FileNotFoundError:
                        logging.error(
                            f'Could not load vector index: {identifier} '
                            'Is the artifact store correctly configured?'
                        )
                        continue
                    except TypeError as e:
                        import traceback

                        logging.error(
                            f'Could not load vector index: {identifier} ' f'{e}'
                        )
                        logging.error(traceback.format_exc())
                        continue
            except exceptions.NotFound:
                pass

    def find_nearest_from_array(
        self,
        h: numpy.typing.ArrayLike,
        component: str,
        vector_index: str,
        n: int = 100,
        within_ids: t.Sequence[str] = (),
    ) -> t.Tuple[t.List[str], t.List[float]]:
        """
        Find the nearest vectors to the given vector.

        :param vector_index: name of vector-index
        :param h: vector
        :param n: number of nearest vectors to return
        :param within_ids: list of ids to search within
        """
        return self[component, vector_index].find_nearest_from_array(
            h, n=n, within_ids=within_ids
        )

    def find_nearest_from_id(
        self,
        id: str,
        component: str,
        vector_index: str,
        n: int = 100,
        within_ids: t.Sequence[str] = (),
    ) -> t.Tuple[t.List[str], t.List[float]]:
        """
        Find the nearest vectors to the given vector.

        :param vector_index: name of vector-index
        :param id: id of the vector to search with
        :param n: number of nearest vectors to return
        :param within_ids: list of ids to search within
        """
        return self[component, vector_index].find_nearest_from_id(
            id, n=n, within_ids=within_ids
        )

    def __getitem__(self, item):
        c = self.db.load(*item)
        if c.uuid not in self.uuid_tool_mapping:
            self.put_component(c)
        return self.get_tool(c.uuid)


class InMemoryVectorSearcher(BaseVectorSearcher):
    """
    Simple hash-set for looking up with vector similarity.

    :param uuid: Unique string identifier of index
    :param dimensions: Dimension of the vector embeddings
    :param measure: measure to assess similarity
    """

    def __init__(
        self,
        identifier: str,
        dimensions: int,
        measure: str = 'cosine',
        component: str = 'VectorIndex',
    ):
        self.identifier = identifier
        self.dimensions = dimensions
        self.component = component

        self._cache: t.Sequence[VectorItem] = []
        self._CACHE_SIZE = 10000

        if isinstance(measure, str):
            self.measure_name = measure
            self.measure = measures[measure]
        else:
            self.measure_name = measure.__name__
            self.measure = measure

        self.h = None
        self.index = None
        self.lookup = None

    def drop(self):
        """Drop the vector index."""

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

    def find_nearest_from_id(self, _id, n=100, within_ids=None):
        """Find the nearest vectors to the given ID.

        :param _id: ID of the vector
        :param n: number of nearest vectors to return
        """
        self.post_create()
        return self.find_nearest_from_array(
            self.h[self.lookup[_id]], n=n, within_ids=within_ids
        )

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

    def initialize(self):
        """Initialize the vector index.

        :param vector_index: Vector index to initialize
        """
        c: VectorIndex = self.db.load(self.component, uuid=self.identifier)
        vectors = c.get_vectors()
        vectors = [
            VectorItem(id=vector['id'], vector=vector['vector']) for vector in vectors
        ]
        self.add(vectors, cache=True)

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
        if not items:
            return
        index = [item.id for item in items]
        h = numpy.stack([item.vector for item in items])

        if self.h is not None:
            old_not_in_new = list(set(self.index) - set(index))
            ix_old = [self.lookup[_id] for _id in old_not_in_new]
            h = numpy.concatenate((self.h[ix_old], h), axis=0)
            index = [self.index[i] for i in ix_old] + index

        out = self._setup(h, index)
        return out

    def delete(self, ids):
        """Delete vectors from the index.

        :param ids: List of IDs to delete
        """
        self.post_create()
        ix = list(map(self.lookup.__getitem__, ids))
        h = numpy.delete(self.h, ix, axis=0)
        index = [_id for _id in self.index if _id not in set(ids)]
        self._setup(h, index)
