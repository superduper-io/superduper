from test.db_config import DBConfig

import pytest

from superduperdb.backends.ibis.field_types import dtype
from superduperdb.backends.ibis.query import Table
from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.document import Document
from superduperdb.components.schema import Schema


@pytest.fixture
def table() -> Table:
    schema = Schema('schema', fields={'id': dtype('str'), 'this': dtype('str')})
    table = Table('documents', schema=schema)
    return table


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_execute_insert_and_find_mongodb(db):
    collection = Collection('documents')
    collection.insert_many([Document({'this': 'is a test'})]).execute(db)
    r = collection.find_one().execute(db)
    assert r['this'] == 'is a test'


@pytest.mark.parametrize("db", [DBConfig.sqldb_empty], indirect=True)
def test_execute_insert_and_find_sqldb(db, table):
    db.add(table)
    table.insert([Document({'this': 'is a test', 'id': '1'})]).execute(db)
    r = table.select('this').limit(1).execute(db).next()
    assert r['this'] == 'is a test'


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_execute_complex_query_mongodb(db):
    collection = Collection('documents')
    collection.insert_many(
        [Document({'this': f'is a test {i}'}) for i in range(100)]
    ).execute(db)

    cur = collection.find().sort('this', -1).limit(10).execute(db)
    expected = [f'is a test {i}' for i in range(99, 89, -1)]
    cur_this = [r['this'] for r in cur]
    assert sorted(cur_this) == sorted(expected)


@pytest.mark.parametrize("db", [DBConfig.sqldb_empty], indirect=True)
def test_execute_complex_query_sqldb(db, table):
    import ibis

    db.add(table)
    table.insert(
        [Document({'this': f'is a test {i}', 'id': str(i)}) for i in range(100)]
    ).execute(db)

    cur = table.select('this').order_by(ibis.desc('this')).limit(10).execute(db)
    expected = [f'is a test {i}' for i in range(99, 89, -1)]
    cur_this = [r['this'] for r in cur]
    assert sorted(cur_this) == sorted(expected)


def test_execute_like_queries_mongodb(db):
    collection = Collection('documents')
    # get a data point for testing
    r = collection.find_one().execute(db)

    out = (
        collection.like({'x': r['x']}, vector_index='test_vector_search', n=10)
        .find()
        .execute(db)
    )

    ids = list(out.scores.keys())
    scores = out.scores

    assert str(r['_id']) in ids[:3]
    assert scores[ids[0]] > 0.999999

    # pre-like
    result = (
        collection.like(Document({'x': r['x']}), vector_index='test_vector_search', n=1)
        .find_one()
        .execute(db)
    )

    assert result['_id'] == r['_id']

    # check queries we didn't have before
    y = collection.distinct('y').execute(db)
    assert set(y) == {0, 1}

    # post-like
    q = collection.find().like(
        Document({'x': r['x']}), vector_index='test_vector_search', n=3
    )
    result = list(q.execute(db))
    assert len(result) == 3
    assert result[0]['_id'] == r['_id']


@pytest.mark.parametrize("db", [DBConfig.sqldb], indirect=True)
def test_execute_like_queries_sqldb(db):
    table = db.load('table', 'documents')
    # get a data point for testing
    r = table.outputs().limit(1).execute(db).next()

    out = (
        table.like({'x': r['x']}, vector_index='test_vector_search', n=10)
        .limit(10)
        .execute(db)
    )
    assert out

    # ids = list(out.scores.keys())
    # scores = out.scores
    #
    # assert str(r['_id']) in ids[:3]
    # assert scores[ids[0]] > 0.999999
    #
    # # pre-like
    # result = (
    #     table.like(Document({'x': r['x']}),
    #                vector_index='test_vector_search', n=1)
    #     .find_one()
    #     .execute(db)
    # )
    #
    # assert result['_id'] == r['_id']
    #
    # # check queries we didn't have before
    # y = table.distinct('y').execute(db)
    # assert set(y) == {0, 1}
    #
    # # post-like
    # q = table.find().like(
    #     Document({'x': r['x']}), vector_index='test_vector_search', n=3
    # )
    # result = list(q.execute(db))
    # assert len(result) == 3
    # assert result[0]['_id'] == r['_id']
