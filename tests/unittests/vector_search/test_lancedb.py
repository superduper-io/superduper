import os
import tempfile
import pandas as pd
import numpy as np

import pytest
from unittest.mock import MagicMock
import pyarrow as pa

from superduperdb import CFG
from superduperdb.vector_search.base import VectorCollectionItem
from superduperdb.misc.config import LanceDB
from superduperdb.vector_search.lancedb_client import (
    LanceDBClient,
    LanceTable,
    LanceVectorIndex,
)


@pytest.fixture(scope="function")
def lance_client():
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, '.test_db')
        config = LanceDB(uri=path)
        client = LanceDBClient(config)
        table = client.create_table(
            "test_table", data=[{'vector': [1.0, 1.0], 'id': 1}], schema=None
        )
        yield client, table


@pytest.fixture(scope="function")
def lance_table(lance_client):
    lance_client, table = lance_client
    yield table


def test_get_table(lance_client):
    lance_client, table = lance_client
    table_name = "test_table"
    measure = "cosine"
    table = lance_client.get_table(table_name, measure)
    assert isinstance(table, LanceTable)
    assert table.table.name == table_name
    assert table.measure == measure


def test_create_table_existing(lance_client):
    lance_client, table = lance_client
    table_name = "test_table"
    data = [{"id": "1", "vector": [1, 2, 3]}, {"id": "2", "vector": [4, 5, 6]}]
    schema = {"id": str, "vector": np.ndarray}
    measure = "euclidean"

    table = lance_client.create_table(table_name, data, schema, measure)
    assert isinstance(table, LanceTable)
    assert table.table.name == table_name
    assert table.measure == measure


def test_create_table_new(lance_client):
    lance_client, table = lance_client
    table_name = "new_table"
    data = [{"id": "1", "vector": [1, 2, 3]}, {"id": "2", "vector": [4, 5, 6]}]
    schema = None
    measure = "euclidean"

    table = lance_client.create_table(table_name, data, schema, measure)

    assert isinstance(table, LanceTable)
    assert table.table.name == table_name
    assert table.measure == measure


@pytest.mark.skip(reason="`tantivy` package needs to be installed")
def test_add(lance_table):
    data = [
        {"id": "1", "vector": [2, 3]},
        {"id": "2", "vector": [5, 6]},
        {"id": "3", "vector": [8, 9]},
    ]
    data = [VectorCollectionItem(**d) for d in data]
    lance_table.add(data)
    assert lance_table.get("1") == data[0]


@pytest.mark.skip(reason="`tantivy` package needs to be installed")
def test_find_nearest_from_id(lance_table):
    identifier = "1"
    limit = 100
    measure = "cosine"
    vector = pd.DataFrame({"vector": [[1, 2]]})
    expected_result = []

    lance_table.where = MagicMock(return_value=vector)
    lance_table.find_nearest_from_array = MagicMock(return_value=expected_result)

    result = lance_table.find_nearest_from_id(identifier, limit, measure)

    assert result == expected_result
    lance_table.where.assert_called_once_with(f"id = '{identifier}'")
    lance_table.find_nearest_from_array.assert_called_once_with(
        vector, limit=limit, measure=measure
    )


def test_find_nearest_from_array(lance_table):
    array = [1, 2]
    limit = 100
    measure = "cosine"
    within_ids = []

    result = lance_table.find_nearest_from_array(array, limit, measure, within_ids)
    assert result[0].id == 1


def test_create_schema():
    dimensions = 3
    expected_schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), dimensions)),
            pa.field("id", pa.string()),
        ]
    )
    vector_index = LanceVectorIndex(config=CFG.vector_search.type)
    schema = vector_index._create_schema(dimensions)
    assert schema.equals(expected_schema)
