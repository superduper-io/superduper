import time
import threading
import traceback
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Sequence,
    Tuple,
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
from superduper.base.datalayer import Datalayer
from superduper.base.metadata import NonExistentMetadataError


class LocalVectorSearchBackend(VectorSearchBackend):
    """Local vector search backend.

    This backend manages multiple vector search indexes locally in memory.
    It loads indexes from the database and creates searcher tools for each index.

    :param searcher_impl: class to use for requesting similarities
    """

    def __init__(self, db: Datalayer, searcher_impl: type[BaseVectorSearcher]) -> None:
        super().__init__()
        self._db = db
        self.searcher_impl = searcher_impl
        # Add lock for thread safety when modifying component mappings
        self._lock = threading.RLock()

    def build_tool(self, component: Any) -> BaseVectorSearcher:
        """Build a searcher tool from a component.

        Creates a new searcher instance for a vector index component.
        This is used when a new vector index is created or loaded.

        :param component: Component to create searcher from
        :return: Initialized searcher instance
        """
        return self.searcher_impl.from_component(self._db, component)

    def initialize(self) -> None:
        """Initialize the vector search by loading indexes from the database.

        This method loads all existing vector indexes from the database
        and initializes them in memory. Called during startup.
        """
        try:
            # Get all vector index identifiers from the database
            for identifier in self._db.show('VectorIndex'):
                try:
                    # Load each vector index from the database
                    vector_index = self._db.load('VectorIndex', identifier=identifier)

                    # Thread-safe registration of the component with the backend
                    with self._lock:
                        self.put_component(vector_index)

                        # Convert raw vector data to VectorItem objects
                        vectors = [VectorItem(**v) for v in vector_index.get_vectors()]

                        # Add vectors to the searcher tool
                        self.get_tool(vector_index.uuid).add(vectors)
                except (FileNotFoundError, TypeError) as e:
                    logging.error(f'Failed to load vector index {identifier}: {e}')
        except NonExistentMetadataError:
            # No vector indexes exist yet - this is normal on first run
            pass

    def find_nearest_from_array(
        self,
        h: ArrayLike,
        vector_index: str,
        n: int = 100,
        within_ids: Sequence[str] = None,
    ) -> Tuple[List[str], List[float]]:
        """Find the nearest vectors to the given vector.

        Delegates to the specific searcher tool for the vector index.

        :param h: vector
        :param vector_index: name of vector-index
        :param n: number of nearest vectors to return
        :param within_ids: list of ids to search within
        :return: Tuple of (ids, scores)
        """
        # No lock needed for read-only operation on a stable reference
        return self[vector_index].find_nearest_from_array(h, n, within_ids)

    def find_nearest_from_id(
        self,
        id: str,
        vector_index: str,
        n: int = 100,
        within_ids: Sequence[str] = None,
    ) -> Tuple[List[str], List[float]]:
        """Find the nearest vectors to the vector identified by ID.

        Delegates to the specific searcher tool for the vector index.

        :param id: id of the vector to search with
        :param vector_index: name of vector-index
        :param n: number of nearest vectors to return
        :param within_ids: list of ids to search within
        :return: Tuple of (ids, scores)
        """
        # No lock needed for read-only operation on a stable reference
        return self[vector_index].find_nearest_from_id(id, n, within_ids)

    def __getitem__(self, identifier: str) -> BaseVectorSearcher:
        """Get a vector index by identifier.

        Loads the vector index if not already cached and returns its searcher tool.
        Provides dict-like access to vector indexes.

        :param identifier: Vector index identifier
        :return: Searcher tool for the vector index
        """
        # Load the vector index component from database
        component = self._db.load('VectorIndex', identifier=identifier)

        # Thread-safe component registration and tool retrieval
        with self._lock:
            # Check if we already have a searcher tool for this component
            if component.uuid not in self.uuid_tool_mapping:
                # Create and register a new searcher tool
                self.put_component(component)

            # Return the searcher tool
            return self.get_tool(component.uuid)


class InMemoryVectorSearcher(BaseVectorSearcher):
    """Simple hash-set for looking up with vector similarity.

    This searcher stores vectors in memory and performs similarity search
    using numpy operations. It supports batching with size and time limits.

    :param identifier: Unique string identifier of index
    :param dimensions: Dimension of the vector embeddings
    :param measure: measure to assess similarity
    """

    # Class constants for batching behavior
    _BATCH_SIZE: Final = 1000  # Maximum items in a batch
    _BATCH_TIMEOUT: Final = 10.0  # Maximum seconds to wait before processing

    def __init__(
        self,
        db: Datalayer,
        identifier: str,
        dimensions: int,
        measure: Union[str, Callable] = 'cosine',
    ) -> None:
        self._db = db
        self.identifier = identifier
        self.dimensions = dimensions

        # Add lock for thread safety
        self._lock = threading.RLock()

        # Batching state
        self._cache: List[VectorItem] = []  # Temporary storage for batch items
        self._last_batch_time = time.time()  # Track when last batch was processed

        # Set up similarity measure
        if isinstance(measure, str):
            self.measure_name = measure
            self.measure = measures[measure]  # Look up predefined measure
        else:
            self.measure_name = measure.__name__
            self.measure = measure

        # Vector storage - Initialize all attributes
        self.vectors: NDArray[np.float64] = None  # Matrix of vectors
        self.index: List[str] = []  # List of vector IDs (parallel to self.h)
        self.lookup: Dict[str, int] = {}  # Map ID to index in self.h

    def initialize(self) -> None:
        """Initialize the vector index from the database.

        Loads all vectors for this index from the database during startup.
        """
        # Load the vector index component
        vector_index = self._db.load('VectorIndex', uuid=self.identifier)

        # Convert raw data to VectorItem objects
        vectors = [VectorItem(**v) for v in vector_index.get_vectors()]

        if vectors:
            # Add vectors using batch mode for efficiency
            self.add(vectors, batch=True)

    def drop(self) -> None:
        """Drop the vector index by clearing all vectors and data from memory."""
        with self._lock:
            self.vectors = None
            self.index = []
            self.lookup = {}
            self._cache = []

    def __len__(self) -> int:
        """Return the number of vectors in the index.

        Includes both committed vectors and those waiting in cache.
        """
        with self._lock:
            committed_count = len(self.index) if self.vectors is not None else 0
            return committed_count + len(self._cache)

    def _normalize(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize vectors for cosine similarity.

        Normalizes vectors to unit length to enable cosine similarity
        via dot product computation.
        """
        # Calculate vector norms (L2)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        # Prevent division by zero for zero vectors
        norms = np.maximum(norms, 1e-10)

        return vectors / norms

    def add(self, items: Sequence[VectorItem], batch: bool = False) -> None:
        """Add vectors to the index.

        In batch mode, accumulates items until size or time limits are reached.
        In non-batch mode, processes items immediately.

        :param items: List of vectors to add
        :param batch: Whether to batch process vectors
        """
        if not items:
            return

        # Non-batch mode: process immediately
        if not batch:
            self._process_items(items)
            return

        # Thread-safe batch processing
        process_now = False
        with self._lock:
            # Batch mode: accumulate items
            self._cache.extend(items)

            # Check if batch should be processed
            batch_size_reached = len(self._cache) >= self._BATCH_SIZE
            batch_timeout_reached = (
                self._cache  # Avoid timeout check if cache is empty
                and time.time() - self._last_batch_time >= self._BATCH_TIMEOUT
            )

            if batch_size_reached or batch_timeout_reached:
                process_now = True

        # Process batch outside of lock if needed
        if process_now:
            self._process_batch()

    def post_create(self) -> None:
        """Process any remaining vectors in cache.

        Called before search operations to ensure all pending vectors
        are indexed and searchable.
        """
        process_needed = False
        with self._lock:
            process_needed = bool(self._cache)

        if process_needed:
            self._process_batch()

    def _process_batch(self) -> None:
        """Process cached items.

        Commits all cached items to the main index and resets batching state.
        """
        with self._lock:
            # No copy needed - we're replacing self._cache with a new empty list
            cached_items = self._cache
            self._cache = []
            self._last_batch_time = time.time()

        # Process items outside the lock
        if cached_items:
            self._process_items(cached_items)

    def _process_items(self, items: Sequence[VectorItem]) -> None:
        """Process vector items and update the index.

        Handles deduplication, normalization, and index updates.

        :param items: Vector items to add
        """
        if not items:
            return

        # Extract data from items (outside lock for efficiency)
        new_ids = [item.id for item in items]
        new_vectors = np.vstack([np.array(item.vector) for item in items])

        with self._lock:
            # Handle existing data
            if self.vectors is not None:
                # Remove duplicates from existing data
                # Keep only IDs that are not being updated
                keep_mask = [
                    i for i, id_ in enumerate(self.index) if id_ not in new_ids
                ]

                if keep_mask:
                    # Combine existing and new data
                    self.vectors = np.vstack((self.vectors[keep_mask], new_vectors))
                    self.index = [self.index[i] for i in keep_mask] + new_ids
                else:
                    # All existing IDs were replaced
                    self.vectors = new_vectors
                    self.index = new_ids
            else:
                # First time adding vectors
                self.vectors = new_vectors
                self.index = new_ids

            # Normalize if using cosine similarity
            if self.measure_name == 'cosine':
                self.vectors = self._normalize(self.vectors)

            # Rebuild lookup table
            self.lookup = {id_: i for i, id_ in enumerate(self.index)}

    def find_nearest_from_array(
        self, h: ArrayLike, n: int = 100, within_ids: Optional[Sequence[str]] = None
    ) -> Tuple[List[str], List[float]]:
        """Find the nearest vectors to the given vector.

        Performs similarity search using the configured measure.
        Optionally restricts search to a subset of IDs.

        :param h: vector
        :param n: number of nearest vectors to return
        :param within_ids: list of IDs to search within
        :return: Tuple of (ids, scores)
        """
        # Ensure all pending items are processed
        self.post_create()

        # Prepare query vector (outside lock for efficiency)
        query = np.array(h).reshape(1, -1)

        with self._lock:
            if self.vectors is None:
                return [], []  # No vectors in index

            if self.measure_name == 'cosine':
                query = self._normalize(query)

            # Handle ID filtering
            if within_ids:
                # Find indices of valid IDs
                indices = [self.lookup[id_] for id_ in within_ids if id_ in self.lookup]
                if not indices:
                    return [], []  # No valid IDs to search

                # Use the vectors at these indices
                target_vectors = self.vectors[indices]  # Returns a view, not a copy
                target_ids = [self.index[i] for i in indices]
            else:
                # Search all vectors
                target_vectors = self.vectors  # Use the original array
                target_ids = self.index  # Use the original list

            # Calculate similarities while still holding the lock
            # This prevents the arrays from changing during computation
            similarities = self.measure(query, target_vectors)[0]

            # Find top N indices (negative for descending sort)
            top_indices = np.argsort(-similarities)[:n]

            # Create results while still holding the lock
            result_ids = [target_ids[i] for i in top_indices]
            result_scores = similarities[top_indices].tolist()

        # Return results (after releasing the lock)
        return result_ids, result_scores

    def find_nearest_from_id(
        self, id: str, n: int = 100, within_ids: Optional[Sequence[str]] = None
    ) -> Tuple[List[str], List[float]]:
        """Find the nearest vectors to the vector identified by ID.

        Convenience method that looks up the vector by ID and then
        calls find_nearest_from_array.

        :param id: id of the vector to search with
        :param n: number of nearest vectors to return
        :param within_ids: list of IDs to search within
        :return: Tuple of (ids, scores)
        """
        # Ensure all pending items are processed
        self.post_create()

        with self._lock:
            # Check if ID exists
            if self.vectors is None or id not in self.lookup:
                return [], []

            # Get vector for ID and use it directly
            # No need for a copy since we'll use it immediately
            vector = self.vectors[self.lookup[id]]

            # We can safely release the lock before calling find_nearest_from_array
            # because we've already made a copy of the vector row we need

        # Search with the vector (outside the lock)
        # This will reacquire the lock as needed
        return self.find_nearest_from_array(vector, n, within_ids)

    def delete(self, ids: Sequence[str]) -> None:
        """Delete vectors from the index.

        Removes specified IDs from the index and rebuilds data structures.

        :param ids: List of IDs to delete
        """
        # Ensure all pending items are processed
        self.post_create()

        with self._lock:
            if not ids or self.vectors is None:
                return

            # Find IDs that actually exist in the index
            ids_to_delete = set(ids) & set(self.lookup.keys())
            if not ids_to_delete:
                return  # Nothing to delete

            # Find indices to keep
            keep_mask = [
                i for i, id_ in enumerate(self.index) if id_ not in ids_to_delete
            ]

            if not keep_mask:
                # All vectors deleted - clear everything
                self.drop()
                return

            # Rebuild data structures without deleted IDs
            self.vectors = self.vectors[keep_mask]
            self.index = [self.index[i] for i in keep_mask]
            self.lookup = {id_: i for i, id_ in enumerate(self.index)}
