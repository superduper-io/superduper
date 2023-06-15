from contextlib import contextmanager
import uuid
import pymilvus
from typing import Iterator, List, Sequence, Dict, Any, Mapping

import numpy

from .base import (
    to_numpy,
    ArrayLike,
    VectorCollection,
    VectorCollectionItem,
    VectorCollectionConfig,
    VectorDatabase,
    VectorCollectionItemId,
    VectorCollectionItemNotFound,
    VectorCollectionResult,
    VectorIndexMeasureType,
    VectorIndexMeasure,
)
from ..misc.config import VectorSearchConfig, MilvusConfig


class MilvusClient:
    def __init__(self, *, config: MilvusConfig) -> None:
        self._config = config
        self._alias = str(uuid.uuid4())
        self._collections: Dict[str, pymilvus.Collection] = {}
        self._indices: Dict[str, pymilvus.Index] = {}

    @contextmanager
    def init(self) -> Iterator["MilvusClient"]:
        """Establish a connection to the Milvus server.

        pymilvus keeps all connections in a global singleton. Each connection has an
        alias which is its unique identifier. We generate a random alias and
        keep track of it to properly close the associated connection.
        """
        pymilvus.connections.connect(
            alias=self._alias,
            host=self._config.host,
            port=self._config.port,
            username=self._config.username,
            password=self._config.password,
            db_name=self._config.db_name,
        )
        try:
            yield self
        finally:
            pymilvus.connections.disconnect(alias=self._alias)
            self._collections.clear()
            self._indices.clear()

    def list_databases(self) -> List[str]:
        return pymilvus.db.list_database(self._alias)

    def list_collections(self) -> List[str]:
        return pymilvus.utility.list_collections(using=self._alias)

    def get_collection(
        self, name: str, *args: Any, **kwargs: Any
    ) -> pymilvus.Collection:
        """Get a `pymilvus.Collection` by name.

        If the collection does not exist, it will be created.

        Note that we keep track of all collections created by this client in order to
        avoid unnecessary introspection calls on every `pymilvus.Collection`
        initialization.
        """
        collection = self._collections.get(name)
        if not collection:
            collection = pymilvus.Collection(
                name,
                *args,
                consistency_level=self._config.consistency_level,
                **kwargs,
                using=self._alias,
            )
        self._collections[name] = collection
        return collection

    def get_index(
        self, collection: pymilvus.Collection, *args: Any, **kwargs: Any
    ) -> pymilvus.Index:
        """Get a `pymilvus.Index` for a collection.

        If the index does not exist, it will be created.

        Here we simplistically assume that each collection has only one index available
        under the default name. This can be revisited later if needed.

        Note that we keep track of all indices created by this client in order to avoid
        unnecessary introspection calls on every `connection.index` call.
        """
        index = self._indices.get(collection.name)
        if not index:
            index = self._get_index(collection, *args, **kwargs)
        self._indices[collection.name] = index
        return index

    def _get_index(
        self, collection: pymilvus.Collection, *args: Any, **kwargs: Any
    ) -> pymilvus.Index:
        try:
            return collection.index(*args, **kwargs)
        except pymilvus.exceptions.IndexNotExistException:
            index = collection.create_index(*args, **kwargs)
            # Milvus keeps all indices in memory. The DB offers tools to load and
            # release indices in run time. Here we load the newly created index right
            # away in order to be able to use it. Later we might need to make this more
            # complex and load indices on demand, although it might be expected to be
            # done by the database itself.
            collection.load()
            return index

    def drop_collection(self, name: str) -> None:
        self._collections.pop(name, None)
        self._indices.pop(name, None)
        pymilvus.utility.drop_collection(name, using=self._alias)

    def drop_all_collections(self) -> None:
        for name in self.list_collections():
            self.drop_collection(name)


class MilvusVectorDatabase(VectorDatabase):
    """

    It is assumed that a column or a field in an upstream database table / collection is
    represented as a single exclusive collection and an index in Milvus. The reason
    behind this is that Milvus works with schemas, but does not allow to
    fully modify them.
    It is much easier to stick to this layout and manage indexes granularly.
    """

    _client: MilvusClient
    _vector_field_name = "vector"

    _measure_type_metric_type_map: Mapping[VectorIndexMeasureType, str] = {
        "l2": "L2",
        "dot": "IP",
    }

    def __init__(self, *, config: VectorSearchConfig) -> None:
        if not config.milvus:
            raise RuntimeError("MilvusConfig is not set")
        super().__init__(config=config)
        self._milvus_config = config.milvus

    @contextmanager
    def init(self) -> Iterator["VectorDatabase"]:
        with MilvusClient(config=self._milvus_config).init() as client:
            self._client = client
            yield self

    def _create_collection_schema(self, dimensions: int) -> pymilvus.CollectionSchema:
        return pymilvus.CollectionSchema(
            fields=[
                pymilvus.FieldSchema(
                    name="id",
                    is_primary=True,
                    dtype=pymilvus.DataType.VARCHAR,
                    max_length=128,
                ),
                pymilvus.FieldSchema(
                    name=self._vector_field_name,
                    dtype=pymilvus.DataType.FLOAT_VECTOR,
                    dim=dimensions,
                ),
            ],
        )

    def _get_collection(self, *, config: VectorCollectionConfig) -> pymilvus.Collection:
        collection = self._client.get_collection(
            name=config.id,
            schema=self._create_collection_schema(dimensions=config.dimensions),
        )
        self._client.get_index(
            collection,
            field_name=self._vector_field_name,
            index_params={
                "metric_type": self._get_metric_type(config.measure),
                "index_type": "HNSW",
                # TODO: use config.parameters here
                "params": {"efConstruction": 128, "M": 32},
            },
        )
        return collection

    def _get_metric_type(self, measure: VectorIndexMeasure) -> str:
        if not isinstance(measure, str):
            raise ValueError(f"Measure functions are not supported: {measure}")
        try:
            return self._measure_type_metric_type_map[measure]
        except KeyError:
            raise ValueError(f"Unsupported measure type: {measure}")

    @contextmanager
    def get_collection(
        self, config: VectorCollectionConfig
    ) -> Iterator[VectorCollection]:
        collection = self._get_collection(config=config)
        with MilvusVectorCollection(collection=collection).init() as vector_collection:
            yield vector_collection


class MilvusVectorCollection(VectorCollection):
    def __init__(self, *, collection: pymilvus.Collection) -> None:
        self._collection = collection

    @contextmanager
    def init(self) -> Iterator["MilvusVectorCollection"]:
        yield self

    def add(self, items: Sequence[VectorCollectionItem]) -> None:
        self._collection.insert(
            [{"id": item.id, "vector": item.vector} for item in items]
        )
        self._collection.flush()

    def find_nearest_from_id(
        self,
        identifier: VectorCollectionItemId,
        *,
        within_ids: Sequence[VectorCollectionItemId] = (),
        limit: int = 100,
        offset: int = 0,
    ) -> List[VectorCollectionResult]:
        if within_ids:
            raise NotImplementedError("within_ids not supported")
        array = self._get_vector_by_id(identifier=identifier)
        return self.find_nearest_from_array(array=array, limit=limit, offset=offset)

    def _get_vector_by_id(self, identifier: VectorCollectionItemId) -> numpy.ndarray:
        record = self._collection.query(
            expr=f'id == "{identifier}"',
            limit=1,
            output_fields=[MilvusVectorDatabase._vector_field_name],
        )
        if not record:
            raise VectorCollectionItemNotFound()
        return record[0][MilvusVectorDatabase._vector_field_name]

    def find_nearest_from_array(
        self,
        array: ArrayLike,
        *,
        within_ids: Sequence[VectorCollectionItemId] = (),
        limit: int = 100,
        offset: int = 0,
    ) -> List[VectorCollectionResult]:
        if within_ids:
            raise NotImplementedError("within_ids not supported")
        result = self._collection.search(
            data=[to_numpy(array)],
            anns_field=MilvusVectorDatabase._vector_field_name,
            param={"metric_type": "L2", "params": {"ef": "top_k"}},
            limit=limit,
            offset=offset,
        )
        out = [
            VectorCollectionResult(id=id_, score=-distance)
            for id_, distance in zip(result[0].ids, result[0].distances)
        ]
        return out
