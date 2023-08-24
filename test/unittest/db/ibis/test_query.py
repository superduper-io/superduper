import ibis
import pytest

from superduperdb.db.ibis.query import IbisConnection, Table


@pytest.fixture
def connection():
    connection = ibis.sqlite.connect("mydb.sqlite")
    # Define the schema of your table

    schema = [('id', 'int64'), ('name', 'string'), ('age', 'int32')]
    schema = ibis.schema({'id': 'int64', 'name': 'string', 'age': 'int32'})
    # Create the table
    table_name = 'my_table'
    connection.create_table(table_name, schema=schema)

    # Insert some sample data into the table
    data_to_insert = [
        (1, 'Alice', 25),
        (2, 'Bob', 26),
        (3, 'Charlie', 27),
        (4, 'Noam', 28),
        (5, 'Chris', 29),
    ]
    connection.insert(table_name, data_to_insert)
    yield connection
    connection.drop_table(table_name)


def test_simple_ibis_query(connection):
    # SuperDuperdb Version
    table = Table(identifier='my_table')

    query = table.select(table.primary_id)
    curr = IbisConnection(connection).execute(query)
    results = [row.unpack() for row in curr]
    assert results == [{'id': 1}, {'id': 2}, {'id': 3}, {'id': 4}, {'id': 5}]


def test_select_ids_ibis_query(connection):
    # SuperDuperdb Version
    table = Table(identifier='my_table')

    query = table.filter(table.age > 24).select_ids()
    curr = IbisConnection(connection).execute(query)
    results = [row.unpack() for row in curr]
    assert results == [{'id': 1}, {'id': 2}, {'id': 3}, {'id': 4}, {'id': 5}]


def test_limit_ibis_query(connection):
    # SuperDuperdb Version
    table = Table(identifier='my_table')

    query = table.filter(table.age > 24).limit(4).select_ids()
    curr = IbisConnection(connection).execute(query)
    results = [row.unpack() for row in curr]
    assert results == [{'id': 1}, {'id': 2}, {'id': 3}, {'id': 4}]
