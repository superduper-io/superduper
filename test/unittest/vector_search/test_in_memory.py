from typing import Iterator

import numpy
import pytest
from superduperdb.vector_search.base import (
    VectorCollectionConfig,
    VectorCollectionItem,
    VectorCollectionItemNotFound,
    VectorDatabase,
)
from superduperdb.vector_search.inmemory import InMemoryVectorDatabase


class TestInMemoryVectorCollection:
    @pytest.fixture
    def manager(self) -> Iterator[VectorDatabase]:
        manager = InMemoryVectorDatabase()
        yield manager

    def test_find_nearest_from_array(self, manager: VectorDatabase) -> None:
        index = manager.get_table(VectorCollectionConfig(id="test", dimensions=1))
        index.add(
            [
                VectorCollectionItem(id=str(i), vector=numpy.array([i * 100]))
                for i in range(100)
            ]
        )

        results = index.find_nearest_from_array(numpy.array([0]), limit=8)
        assert len(results) == 8
        ids = [int(r.id) for r in results]
        assert all(i <= 15 for i in ids)

    def test_find_nearest_from_id(self, manager: VectorDatabase) -> None:
        index = manager.get_table(VectorCollectionConfig(id="test", dimensions=1))
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

    def test_find_nearest_from_array__limit_offset(
        self, manager: VectorDatabase
    ) -> None:
        index = manager.get_table(VectorCollectionConfig(id="test", dimensions=1))
        index.add(
            [
                VectorCollectionItem(id=str(i), vector=numpy.array([i * 100]))
                for i in range(100)
            ]
        )

        for offset in range(10):
            results = index.find_nearest_from_array(
                numpy.array([0]), limit=1, offset=offset
            )
            assert len(results) == 1
            ids = [int(r.id) for r in results]
            assert ids == [offset]

    def test_find_nearest_from_id__not_found(self, manager: VectorDatabase) -> None:
        index = manager.get_table(VectorCollectionConfig(id="test", dimensions=1))
        with pytest.raises(VectorCollectionItemNotFound):
            index.find_nearest_from_id("15")

    def test_add__overwrite(self, manager: VectorDatabase) -> None:
        index = manager.get_table(VectorCollectionConfig(id="test", dimensions=1))
        index.add([VectorCollectionItem(id="0", vector=numpy.array([0]))])
        index.add([VectorCollectionItem(id="1", vector=numpy.array([1]))])

        index.add([VectorCollectionItem(id="1", vector=numpy.array([100]))])

        results = index.find_nearest_from_array(numpy.array([99]), limit=1)
        ids = [int(r.id) for r in results]
        assert ids == [1]
