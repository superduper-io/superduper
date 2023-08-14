import ibis
import pandas as pd
import random
import pytest


from superduperdb.ext.numpy.array import array
from superduperdb.misc import superduper


def random_str():
    return ''.join([chr(random.randint(97, 122)) for _ in range(10)])


@pytest.fixture()
def table():

    e = array(' float32', (32,))

    conn = ibis.sqlite.connect('test/material/ibis_test.sqlite')
    # db = superduper(conn)

    df = pd.DataFrame({
        'x': [random.random() for _ in range(100)], 'y': [random_str() for _ in range(100)]
    })
    conn.create_table('test', df)
    yield conn.table('test')
    for table in conn.list_tables():
        conn.drop_table(table)


def test_database(table):
    breakpoint()
