import traceback
import typing as t
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

from superduper import logging
from superduper.backends.base.vector_search import (
    BaseVectorSearcher,
    VectorItem,
    VectorSearchBackend,
    measures,
)
from superduper.base.exceptions import InvalidArguments
from superduper.base.metadata import NonExistentMetadataError

if t.TYPE_CHECKING:
    from superduper import VectorIndex
    from superduper.backends.base.backends import Bookkeeping

T = TypeVar('T')  # Generic type for callable return values


class LocalVectorSearchBackend(VectorSearchBackend):
    """Local vector search backend.

    :param searcher_impl: class to use for requesting similarities
    """

    def __init__(self, searcher_impl: type[BaseVectorSearcher]) -> None:
        # Explicit call to avoid untyped super().__init__
        VectorSearchBackend.__init__(self)

        self.searcher_impl = searcher_impl
        self._db: Optional['Bookkeeping'] = None

    @property
    def db(self) -> Optional['Bookkeeping']:
        """Get the database reference.

        :return: Current database instance
        """
        return self._db

    @db.setter
    def db(self, value: 'Bookkeeping') -> None:
        """Set the database reference and update all tools.

        :param value: Database instance to set
        """
        logging.warn("Replacing database")
        self._db = value
        for tool in self.tools:
            tool.db = value
        return None

    def build_tool(self, component: Any) -> BaseVectorSearcher:
        """Build a searcher tool from a component.

        :param component: Component to create searcher from
        :return: Initialized searcher instance
        """
        searcher = self.searcher_impl.from_component(component)
        return cast(BaseVectorSearcher, searcher)  # Cast to avoid Any return type

    def initialize(self) -> None:
        """Initialize the vector search by loading indexes from the database."""
        if self._db is None:
            logging.warn("Database not set during initialization")
            return None

        db = self._db  # Type narrowing for mypy
        try:
            for identifier in db.show('VectorIndex'):
                try:
                    vector_index: 'VectorIndex' = db.load(
                        'VectorIndex', identifier=identifier
                    )
                    self.put_component(vector_index)
                    vector_items = [
                        VectorItem(**vector) for vector in vector_index.get_vectors()
                    ]
                    self.get_tool(vector_index.uuid).add(vector_items)
                except FileNotFoundError:
                    logging.error(
                        f'Could not load vector index: {identifier} '
                        'Is the artifact store correctly configured?'
                    )
                except TypeError as e:
                    logging.error(f'Could not load vector index: {identifier} {e}')
                    logging.error(traceback.format_exc())
        except NonExistentMetadataError:
            pass  # No vector indexes exist yet

        return None

    def find_nearest_from_array(
        self,
        h: ArrayLike,
        vector_index: str,
        n: int = 100,
        within_ids: Sequence[str] = (),
    ) -> Tuple[List[str], List[float]]:
        """Find the nearest vectors to the given vector.

        :param h: vector
        :param vector_index: name of vector-index
        :param n: number of nearest vectors to return
        :param within_ids: list of ids to search within
        :return: Tuple of (ids, scores)
        """
        return self[vector_index].find_nearest_from_array(h, n=n, within_ids=within_ids)

    def find_nearest_from_id(
        self,
        id: str,
        vector_index: str,
        n: int = 100,
        within_ids: Sequence[str] = (),
    ) -> Tuple[List[str], List[float]]:
        """Find the nearest vectors to the vector identified by ID.

        :param id: id of the vector to search with
        :param vector_index: name of vector-index
        :param n: number of nearest vectors to return
        :param within_ids: list of ids to search within
        :return: Tuple of (ids, scores)
        """
        return self[vector_index].find_nearest_from_id(id, n=n, within_ids=within_ids)

    def __getitem__(self, identifier: str) -> BaseVectorSearcher:
        """Get a vector index by identifier.

        :param identifier: Vector index identifier
        :return: Searcher tool for the vector index
        """
        if self._db is None:
            raise ValueError("Database not set")

        db = self._db  # Type narrowing for mypy
        component = db.load('VectorIndex', identifier=identifier)
        if component.uuid not in self.uuid_tool_mapping:
            self.put_component(component)
        searcher = self.get_tool(component.uuid)
        return cast(BaseVectorSearcher, searcher)  # Cast to avoid Any return


class InMemoryVectorSearcher(BaseVectorSearcher):
    """Simple hash-set for looking up with vector similarity.

    :param identifier: Unique string identifier of index
    :param dimensions: Dimension of the vector embeddings
    :param measure: measure to assess similarity
    """

    _CACHE_SIZE: Final[int] = 10000

    def __init__(
        self,
        identifier: str,
        dimensions: int,
        measure: Union[
            str,
            Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
        ] = 'cosine',
    ) -> None:
        self.identifier = identifier
        self.dimensions = dimensions
        self._cache: List[VectorItem] = []

        if isinstance(measure, str):
            self.measure_name = measure
            self.measure = measures[measure]
        else:
            self.measure_name = measure.__name__
            self.measure = measure

        self.h: Optional[NDArray[np.float64]] = None
        self.index: Optional[List[str]] = None
        self.lookup: Optional[Dict[str, int]] = None

    def drop(self) -> None:
        """Drop the vector index by clearing all vectors and data from memory."""
        self.h = None
        self.index = None
        self.lookup = None
        self._cache = []
        return None

    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return 0 if self.h is None else self.h.shape[0]

    def to_numpy(self, vector: ArrayLike) -> NDArray[np.float64]:
        """Convert a vector to numpy array."""
        return (
            vector
            if isinstance(vector, np.ndarray)
            else np.array(vector, dtype=np.float64)
        )

    def _setup(self, h: NDArray[np.float64], index: List[str]) -> None:
        """Set up the vector index.

        :param h: Array of vectors
        :param index: List of vector IDs
        """
        h = self.to_numpy(h)

        if self.measure_name == 'cosine':
            # Normalize vectors for cosine similarity
            norms = np.linalg.norm(h, axis=1)[:, None]
            # Prevent division by zero
            norms = np.maximum(norms, 1e-10)
            h = h / norms

        self.h = h
        self.index = index
        self.lookup = dict(zip(index, range(len(index))))
        return None

    def find_nearest_from_id(
        self, _id: str, n: int = 100, within_ids: Optional[Sequence[str]] = None
    ) -> Tuple[List[str], List[float]]:
        """Find the nearest vectors to the given ID.

        :param _id: ID of the vector
        :param n: number of nearest vectors to return
        :param within_ids: list of IDs to search within
        :return: Tuple of (ids, scores)
        """
        self.post_create()

        if (
            self.h is None
            or self.index is None
            or self.lookup is None
            or _id not in self.lookup
        ):
            logging.error(f"Vector with ID {_id} not found or index not initialized")
            return [], []

        return self.find_nearest_from_array(
            self.h[self.lookup[_id]], n=n, within_ids=within_ids
        )

    def find_nearest_from_array(
        self, h: ArrayLike, n: int = 100, within_ids: Optional[Sequence[str]] = None
    ) -> Tuple[List[str], List[float]]:
        """Find the nearest vectors to the given vector.

        :param h: vector
        :param n: number of nearest vectors to return
        :param within_ids: list of IDs to search within
        :return: Tuple of (ids, scores)
        """
        self.post_create()

        if self.h is None or self.index is None or self.lookup is None:
            logging.error('Vector database not initialized')
            return [], []

        # Convert and normalize input vector
        h_array = self.to_numpy(h).reshape(1, -1)
        if self.measure_name == 'cosine' and np.linalg.norm(h_array) > 0:
            h_array = h_array / np.linalg.norm(h_array)

        # Calculate similarities
        if within_ids:
            # Filter IDs that exist in the lookup
            valid_ids = [id_ for id_ in within_ids if id_ in self.lookup]
            if not valid_ids:
                return [], []

            ix = [self.lookup[id_] for id_ in valid_ids]
            similarities = self.measure(h_array, self.h[ix, :])

            # Get top scores
            top_n_idxs = np.argsort(-similarities[0])[: min(n, len(ix))]
            result_indices = [ix[i] for i in top_n_idxs]
            scores = similarities[0][top_n_idxs].tolist()
        else:
            similarities = self.measure(h_array, self.h)[0]

            # Get top scores
            sorted_indices = np.argsort(-similarities)[: min(n, len(self.index))]
            result_indices = sorted_indices.tolist()
            scores = similarities[sorted_indices].tolist()

        if self.index is None:  # Extra check for mypy
            return [], []

        result_ids = [self.index[i] for i in result_indices]
        return result_ids, scores

    def initialize(self) -> None:
        """Initialize the vector index from the database."""
        if not hasattr(self, 'db') or self.db is None:
            logging.error("Database not set during initialization")
            return None

        db = self.db  # Type narrowing for mypy
        component = db.load('VectorIndex', uuid=self.identifier)
        vectors = component.get_vectors()
        vector_items = [
            VectorItem(id=vector['id'], vector=vector['vector']) for vector in vectors
        ]
        self.add(vector_items, batch=True)
        return None

    def add(self, items: Sequence[VectorItem] = (), batch: bool = False) -> None:
        """Add vectors to the index.

        :param items: List of vectors to add
        :param batch: Whether to batch process vectors
        """
        if not items:
            raise InvalidArguments("empty items")

        if not batch:
            self._add(items)
            return None

        # Add to cache and process when full
        self._cache.extend(items)
        if len(self._cache) >= self._CACHE_SIZE:
            self._add(self._cache)
            self._cache = []

        return None

    def post_create(self) -> None:
        """Process any remaining vectors in cache."""
        if self._cache:
            self._add(self._cache)
            self._cache = []
        return None

    def _add(self, items: Sequence[VectorItem]) -> None:
        """Add vectors directly to the index.

        :param items: Vector items to add
        """
        if not items:
            return None

        # Extract vectors and IDs
        index = [item.id for item in items]
        vectors = [self.to_numpy(item.vector) for item in items]

        # Using np.vstack which is statically typed instead of np.stack
        h: NDArray[np.float64] = np.vstack(vectors)

        # If we already have vectors, preserve the existing ones
        if self.h is not None and self.index is not None and self.lookup is not None:
            old_not_in_new = list(set(self.index) - set(index))
            if old_not_in_new:
                ix_old = [self.lookup[_id] for _id in old_not_in_new]

                # Using np.vstack which is statically typed instead of np.concatenate
                h = np.vstack((self.h[ix_old], h))
                index = [self.index[i] for i in ix_old] + index

        self._setup(h, index)
        return None

    def delete(self, ids: Sequence[str]) -> None:
        """Delete vectors from the index.

        :param ids: List of IDs to delete
        """
        self.post_create()

        if not ids or self.h is None or self.index is None or self.lookup is None:
            return None

        # Filter for IDs that exist in the index
        ids_to_delete = [id_ for id_ in ids if id_ in self.lookup]
        if not ids_to_delete:
            return None

        # Delete vectors and rebuild index
        ix = [self.lookup[id_] for id_ in ids_to_delete]
        h = np.delete(self.h, ix, axis=0)
        index = [_id for _id in self.index if _id not in set(ids_to_delete)]
        self._setup(h, index)
        return None
