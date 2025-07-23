import re
import typing as t
import uuid
from copy import deepcopy

import numpy as np
from qdrant_client import QdrantClient, models
from superduper import CFG, logging
from superduper.backends.base.vector_search import (
    BaseVectorSearcher,
    VectorIndexMeasureType,
    VectorItem,
)
from superduper.misc.retry import Retry

# Import gRPC exceptions if available
try:
    import grpc
    from qdrant_client.http.exceptions import (
        ResponseHandlingException,
        UnexpectedResponse,
    )

    # Common exceptions to retry on
    QDRANT_RETRY_EXCEPTIONS = [
        grpc.RpcError,
        ResponseHandlingException,
        UnexpectedResponse,
        ConnectionError,
        TimeoutError,
    ]

    # Try to add the InactiveRpcError if available
    try:
        QDRANT_RETRY_EXCEPTIONS.append(grpc._channel._InactiveRpcError)
    except AttributeError:
        pass

    QDRANT_RETRY_EXCEPTIONS = tuple(QDRANT_RETRY_EXCEPTIONS)
except ImportError:
    # Fallback if gRPC or Qdrant exceptions are not available
    QDRANT_RETRY_EXCEPTIONS = (ConnectionError, TimeoutError)

ID_PAYLOAD_KEY = "_id"


class QdrantVectorSearcher(BaseVectorSearcher):
    """
    Implementation of a vector index using [Qdrant](https://qdrant.tech/).

    :param identifier: Unique string identifier of index
    :param dimensions: Dimension of the vector embeddings
    :param h: Seed vectors ``numpy.ndarray``
    :param index: list of IDs
    :param measure: measure to assess similarity
    :param batch_size: Number of vectors to upsert in a single batch (default: 512)
    """

    def __init__(
        self,
        identifier: str,
        dimensions: int,
        measure: t.Optional[str] = None,
        component: str = 'VectorIndex',
        batch_size: int = 512,
    ):
        config_dict = deepcopy(CFG.vector_search_kwargs)
        try:
            plugin, uri = CFG.vector_search_engine.split("://")
            if not uri.startswith("http") and uri != ":memory:":
                uri = f"http://{uri}"
            if uri:
                config_dict['location'] = uri
        except ValueError as e:
            if 'not enough values to unpack' in str(e):
                plugin = CFG.vector_search_engine
            else:
                raise e

        assert (
            plugin == "qdrant"
        ), "Only 'qdrant' vector search engine is supported in QdrantVectorSearcher."

        # Use an in-memory instance by default
        # https://github.com/qdrant/qdrant-client#local-mode
        config_dict = config_dict or {"location": ":memory:"}
        if '6334' in config_dict['location']:
            config_dict["prefer_grpc"] = True
            # Set a longer timeout for gRPC connections to avoid DEADLINE_EXCEEDED
            if "timeout" not in config_dict:
                config_dict["timeout"] = 60  # 60 seconds timeout
        self.client = QdrantClient(**config_dict)
        self.identifier = identifier
        self.measure = measure
        self.batch_size = batch_size

        self.identifier = re.sub("\W+", "", identifier)
        if not self.client.collection_exists(self.identifier):
            measure = (
                measure.name if isinstance(measure, VectorIndexMeasureType) else measure
            )
            distance = self._distance_mapping(measure)
            self.client.create_collection(
                collection_name=self.identifier,
                vectors_config=models.VectorParams(size=dimensions, distance=distance),
            )
        self.component = component

    def initialize(self):
        """Initialize the vector index.

        :param db: Datalayer instance
        """
        pass

    def __len__(self):
        return self.client.get_collection(self.identifier).vectors_count

    def _create_collection(self):
        measure = (
            self.measure.name
            if isinstance(self.measure, VectorIndexMeasureType)
            else self.measure
        )
        distance = self._distance_mapping(measure)
        self.client.create_collection(
            collection_name=self.identifier,
            vectors_config=models.VectorParams(size=self.dimensions, distance=distance),
        )

    def add(self, items: t.Sequence[VectorItem], cache: bool = False) -> None:
        """Add vectors to the index.

        :param items: List of vectors to add
        :param cache: Cache vectors (not used in Qdrant implementation).
        """
        if not items:
            return
        points = []
        for item in items:
            if hasattr(item.vector, "tolist"):
                vector = item.vector.tolist()
            else:
                vector = item.vector
            point = models.PointStruct(
                id=self._convert_id(item.id),
                vector=vector,
                payload={ID_PAYLOAD_KEY: item.id},
            )
            points.append(point)

        total_points = len(points)
        total_batches = (total_points + self.batch_size - 1) // self.batch_size

        logging.info(
            f"Adding {total_points} points to Qdrant index '{self.identifier}' "
            f"in {total_batches} batches (batch_size={self.batch_size})"
        )

        # Process points in batches with retry logic
        for batch_idx, i in enumerate(range(0, total_points, self.batch_size), 1):
            batch = points[i : i + self.batch_size]
            batch_size = len(batch)

            logging.info(
                f"Processing batch {batch_idx}/{total_batches} "
                f"({batch_size} points, {i}/{total_points} completed)"
            )

            self._upsert_batch_with_retry(batch, batch_idx, total_batches)

        logging.info(
            f"✓ Successfully added all {total_points} points to Qdrant index "
            f"'{self.identifier}' in {total_batches} batches"
        )

    def drop(self):
        """Drop the vector index."""
        if self.client.collection_exists(self.identifier):
            self.client.delete_collection(self.identifier)

    def delete(self, ids: t.Sequence[str]) -> None:
        """Delete vectors from the index.

        :param ids: List of IDs to delete
        """
        self.client.delete(
            collection_name=self.identifier,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key=ID_PAYLOAD_KEY, match=models.MatchAny(any=list(ids))
                    )
                ]
            ),
        )

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
        return self._query_nearest(_id, n, within_ids)

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
        return self._query_nearest(h, n, within_ids)

    def _query_nearest(
        self,
        query: t.Union[np.typing.ArrayLike, str],
        n: int = 100,
        within_ids: t.Sequence[str] = (),
    ) -> t.Tuple[t.List[str], t.List[float]]:
        query_filter = None
        if within_ids:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=ID_PAYLOAD_KEY, match=models.MatchAny(any=list(within_ids))
                    )
                ]
            )

        search_result = self.client.query_points(
            collection_name=self.identifier,
            query=query,
            limit=n,
            query_filter=query_filter,
            with_payload=[ID_PAYLOAD_KEY],
            using=None,
        ).points

        ids = [hit.payload[ID_PAYLOAD_KEY] for hit in search_result if hit.payload]
        scores = [hit.score for hit in search_result]

        return ids, scores

    def _distance_mapping(self, measure: t.Optional[str] = None) -> models.Distance:
        if measure == "cosine":
            return models.Distance.COSINE
        if measure == "l2":
            return models.Distance.EUCLID
        if measure == "dot":
            return models.Distance.DOT
        else:
            raise ValueError(f"Unsupported measure: {measure}")

    def _convert_id(self, _id: str) -> str:
        """
        Converts any string into a UUID string based on a seed.

        Qdrant accepts UUID strings and unsigned integers as point ID.
        We use a seed to convert each string into a UUID string deterministically.
        This allows us to overwrite the same point with the original ID.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, _id))

    def _upsert_batch_with_retry(
        self, batch: t.List[models.PointStruct], batch_idx: int, total_batches: int
    ) -> None:
        """
        Upsert a batch of points with retry logic.

        :param batch: List of PointStruct objects to upsert
        :param batch_idx: Current batch index (1-based)
        :param total_batches: Total number of batches
        """
        # Create a retry decorator instance for Qdrant-specific exceptions
        retry = Retry(exception_types=QDRANT_RETRY_EXCEPTIONS)

        @retry
        def _do_upsert():
            self.client.upsert(collection_name=self.identifier, points=batch)
            logging.debug(
                f"Successfully upserted batch {batch_idx}/{total_batches} "
                f"({len(batch)} points)"
            )

        try:
            _do_upsert()
            logging.info(f"✓ Batch {batch_idx}/{total_batches} completed successfully")
        except Exception as e:
            logging.error(
                f"✗ Failed to upsert batch {batch_idx}/{total_batches} "
                f"after retries: {e}"
            )
            raise
