import re
import typing as t
import uuid
from copy import deepcopy

import numpy as np
from qdrant_client import QdrantClient, models
from superduper import CFG
from superduper.backends.base.vector_search import (
    BaseVectorSearcher,
    VectorIndexMeasureType,
    VectorItem,
)

ID_PAYLOAD_KEY = "_id"


class QdrantVectorSearcher(BaseVectorSearcher):
    """
    Implementation of a vector index using [Qdrant](https://qdrant.tech/).

    :param identifier: Unique string identifier of index
    :param dimensions: Dimension of the vector embeddings
    :param h: Seed vectors ``numpy.ndarray``
    :param index: list of IDs
    :param measure: measure to assess similarity
    """

    def __init__(
        self,
        uuid: str,
        dimensions: int,
        measure: t.Optional[str] = None,
    ):
        config_dict = deepcopy(CFG.vector_search_kwargs)
        self.vector_name: t.Optional[str] = config_dict.pop("vector_name", None)
        # Use an in-memory instance by default
        # https://github.com/qdrant/qdrant-client#local-mode
        config_dict = config_dict or {"location": ":memory:"}
        self.client = QdrantClient(**config_dict)
        self.collection_name = uuid
        self.measure = measure

        self.collection_name = re.sub("\W+", "", uuid)
        if not self.client.collection_exists(self.collection_name):
            measure = (
                measure.name if isinstance(measure, VectorIndexMeasureType) else measure
            )
            distance = self._distance_mapping(measure)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=dimensions, distance=distance),
            )

    def initialize(self, db):
        """Initialize the vector index.

        :param db: Datalayer instance
        """
        pass

    def __len__(self):
        return self.client.get_collection(self.collection_name).vectors_count

    def _create_collection(self):
        measure = (
            self.measure.name
            if isinstance(self.measure, VectorIndexMeasureType)
            else self.measure
        )
        distance = self._distance_mapping(measure)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.dimensions, distance=distance),
        )

    def add(self, items: t.Sequence[VectorItem], cache: bool = False) -> None:
        """Add vectors to the index.

        :param items: List of vectors to add
        :param cache: Cache vectors (not used in Qdrant implementation).
        """
        points = []
        for item in items:
            point = models.PointStruct(
                id=self._convert_id(item.id),
                vector=(
                    {self.vector_name: item.vector.tolist()}
                    if self.vector_name
                    else item.vector.tolist()
                ),
                payload={ID_PAYLOAD_KEY: item.id},
            )
            points.append(point)
        self.client.upsert(collection_name=self.collection_name, points=points)

    def delete(self, ids: t.Sequence[str]) -> None:
        """Delete vectors from the index.

        :param ids: List of IDs to delete
        """
        self.client.delete(
            collection_name=self.collection_name,
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
            collection_name=self.collection_name,
            query=query,
            limit=n,
            query_filter=query_filter,
            with_payload=[ID_PAYLOAD_KEY],
            using=self.vector_name,
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
