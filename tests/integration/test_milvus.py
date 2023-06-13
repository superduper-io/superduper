import pytest
import numpy
from typing import Iterator

from superduperdb.misc.config import VectorSearchConfig, MilvusConfig
from superduperdb.vector_search.base import (
    VectorIndexItem,
    VectorIndexManager,
    VectorIndexItemNotFound,
)
from superduperdb.vector_search.milvus import MilvusClient, MilvusVectorIndexManager


class TestMilvusClient:
    def test_list_databases(self, milvus_client: MilvusClient) -> None:
        dbs = milvus_client.list_databases()
        assert dbs == ["default"]


class TestMilvusVectorIndex:
    @pytest.fixture
    def manager(self, milvus_server: MilvusConfig) -> Iterator[VectorIndexManager]:
        with MilvusVectorIndexManager(
            config=VectorSearchConfig(milvus=milvus_server)
        ).init() as manager:
            yield manager

    def test_find_nearest_from_array(self, manager: MilvusVectorIndexManager) -> None:
        with manager.get_index("test", dimensions=1) as index:
            index.add(
                [
                    VectorIndexItem(id=str(i), vector=numpy.array([i]))
                    for i in range(100)
                ]
            )

            results = index.find_nearest_from_array(numpy.array([15]), limit=8)
            assert len(results) == 8
            ids = [int(r.id) for r in results]
            assert all(5 <= i <= 25 for i in ids)

    def test_find_nearest_from_id(self, manager: MilvusVectorIndexManager) -> None:
        with manager.get_index("test", dimensions=1) as index:
            index.add(
                [
                    VectorIndexItem(id=str(i), vector=numpy.array([i]))
                    for i in range(100)
                ]
            )

            results = index.find_nearest_from_id("15", limit=8)
            assert len(results) == 8
            ids = [int(r.id) for r in results]
            assert all(5 <= i <= 25 for i in ids)

    def test_find_nearest_from_id__not_found(
        self, manager: MilvusVectorIndexManager
    ) -> None:
        with manager.get_index("test", dimensions=1) as index:
            with pytest.raises(VectorIndexItemNotFound):
                index.find_nearest_from_id("15")

    def test_add__overwrite(self, manager: MilvusVectorIndexManager) -> None:
        with manager.get_index("test", dimensions=1) as index:
            index.add([VectorIndexItem(id="0", vector=numpy.array([0]))])
            index.add([VectorIndexItem(id="1", vector=numpy.array([1]))])

            index.add([VectorIndexItem(id="1", vector=numpy.array([100]))])

            results = index.find_nearest_from_array(numpy.array([99]), limit=1)
            ids = [int(r.id) for r in results]
            assert ids == [1]
