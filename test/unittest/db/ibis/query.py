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
        (2, 'Alice', 25),
        (3, 'Alice', 25),
        (4, 'Alice', 25),
        (5, 'Alice', 25),
        (6, 'Alice', 25),
        (7, 'Alice', 25),
        (8, 'Bob', 30),
        (9, 'Charlie', 28),
    ]
    connection.insert(table_name, data_to_insert)
    yield connection
    connection.drop_table(table_name)


def test_simple_ibis_query(connection):
    # SuperDuperdb Version
    table = Table(name='my_table')
    specific_ids = [1, 2]

    query = table.filter(table.age == 25).select_from_ids(specific_ids)
    curr = IbisConnection(connection).execute(query)
    output = curr.to_dict()
    assert output == {
        'id': {0: 1, 1: 2},
        'name': {0: 'Alice', 1: 'Alice'},
        'age': {0: 25, 1: 25},
    }
