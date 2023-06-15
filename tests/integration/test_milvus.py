import pytest
import numpy
from typing import Iterator

from superduperdb.misc.config import VectorSearchConfig, MilvusConfig
from superduperdb.vector_search.base import (
    VectorCollectionItem,
    VectorDatabase,
    VectorCollectionItemNotFound,
    VectorCollectionConfig,
)
from superduperdb.vector_search.milvus import MilvusClient, MilvusVectorDatabase


@pytest.mark.skip(reason='See issue #291')
class TestMilvusClient:
    def test_list_databases(self, milvus_client: MilvusClient) -> None:
        dbs = milvus_client.list_databases()
        assert dbs == ["default"]


@pytest.mark.skip(reason='See issue #291')
class TestMilvusVectorCollection:
    @pytest.fixture
    def manager(self, milvus_server: MilvusConfig) -> Iterator[VectorDatabase]:
        with MilvusVectorDatabase(
            config=VectorSearchConfig(milvus=milvus_server)
        ).init() as manager:
            yield manager

    def test_find_nearest_from_array(self, manager: VectorDatabase) -> None:
        with manager.get_collection(
            VectorCollectionConfig(id="test", dimensions=1)
        ) as index:
            index.add(
                [
                    VectorCollectionItem(id=str(i), vector=numpy.array([i]))
                    for i in range(100)
                ]
            )

            results = index.find_nearest_from_array(numpy.array([15]), limit=8)
            assert len(results) == 8
            ids = [int(r.id) for r in results]
            assert all(5 <= i <= 25 for i in ids)

    def test_find_nearest_from_id(self, manager: VectorDatabase) -> None:
        with manager.get_collection(
            VectorCollectionConfig(id="test", dimensions=1)
        ) as index:
            index.add(
                [
                    VectorCollectionItem(id=str(i), vector=numpy.array([i]))
                    for i in range(100)
                ]
            )

            results = index.find_nearest_from_id("15", limit=8)
            assert len(results) == 8
            ids = [int(r.id) for r in results]
            assert all(5 <= i <= 25 for i in ids)

    def test_find_nearest_from_id__not_found(self, manager: VectorDatabase) -> None:
        with manager.get_collection(
            VectorCollectionConfig(id="test", dimensions=1)
        ) as index:
            with pytest.raises(VectorCollectionItemNotFound):
                index.find_nearest_from_id("15")

    def test_add__overwrite(self, manager: VectorDatabase) -> None:
        with manager.get_collection(
            VectorCollectionConfig(id="test", dimensions=1)
        ) as index:
            index.add([VectorCollectionItem(id="0", vector=numpy.array([0]))])
            index.add([VectorCollectionItem(id="1", vector=numpy.array([1]))])

            index.add([VectorCollectionItem(id="1", vector=numpy.array([100]))])

            results = index.find_nearest_from_array(numpy.array([99]), limit=1)
            ids = [int(r.id) for r in results]
            assert ids == [1]
