from test.db_config import DBConfig

import lorem
import pytest

from superduperdb.backends.ibis.field_types import dtype
from superduperdb.backends.mongodb.query import MongoQuery
from superduperdb.base.document import Document
from superduperdb.components.schema import Schema
from superduperdb.components.table import Table


@pytest.fixture
def table():
    schema = Schema('schema', fields={'id': dtype('str'), 'this': dtype('str')})
    table = Table('documents', schema=schema)
    return table


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_execute_insert_and_find_mongodb(db):
    t = db['documents']
    insert_query = t.insert_many([Document({'this': 'is a test'})])
    db.execute(insert_query)
    select_query = t.find_one()
    r = db.execute(select_query)
    assert r['this'] == 'is a test'


@pytest.mark.parametrize("db", [DBConfig.sqldb_empty], indirect=True)
def test_execute_insert_and_find_sqldb(db, table):
    db.add(table)
    db[table.identifier].insert([Document({'this': 'is a test', 'id': '1'})]).execute(
        db
    )
    r = db[table.identifier].select('this').limit(1).execute(db).next()
    assert r['this'] == 'is a test'


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_execute_complex_query_mongodb(db):
    t = db['documents']

    insert_query = t.insert_many(
        [Document({'this': f'is a test {i}'}) for i in range(100)]
    )
    db.execute(insert_query)

    select_query = t.find().sort('this', -1).limit(10)
    cur = db.execute(select_query)

    expected = [f'is a test {i}' for i in range(99, 89, -1)]
    cur_this = [r['this'] for r in cur]
    assert sorted(cur_this) == sorted(expected)


@pytest.mark.parametrize("db", [DBConfig.sqldb_empty], indirect=True)
def test_execute_complex_query_sqldb(db, table):
    import ibis

    db.add(table)
    table = db[table.identifier]
    table.insert(
        [Document({'this': f'is a test {i}', 'id': str(i)}) for i in range(100)]
    ).execute(db)

    cur = table.select('this').order_by(ibis.desc('this')).limit(10).execute(db)
    expected = [f'is a test {i}' for i in range(99, 89, -1)]
    cur_this = [r['this'] for r in cur]
    assert sorted(cur_this) == sorted(expected)


def test_execute_like_queries_mongodb(db):
    collection = MongoQuery(identifier='documents', db=db)
    # Get a data point for testing
    r = collection.find_one({}).do_execute(db)

    out = (
        collection.like({'x': r['x']}, vector_index='test_vector_search', n=10)
        .find()
        .do_execute(db)
    )
    scores = out.scores
    ids = [o['_id'] for o in list(out)]
    from bson import ObjectId

    assert ObjectId(r['_id']) in ids
    assert scores[str(ids[0])] > 0.999999

    # Pre-like
    result = (
        collection.like(Document({'x': r['x']}), vector_index='test_vector_search', n=1)
        .find_one()
        .do_execute(db)
    )

    assert result['_id'] == ObjectId(r['_id'])

    # Post-like
    q = collection.find({}).like(
        Document({'x': r['x']}), vector_index='test_vector_search', n=3
    )
    result = q.do_execute(db)
    result = list(result)
    assert len(result) == 3
    assert result[0]['_id'] == ObjectId(r['_id'])


@pytest.mark.parametrize("db", [DBConfig.sqldb], indirect=True)
def test_execute_like_queries_sqldb(db):
    table = db.load('table', 'documents')
    # get a data point for testing
    table = db[table.identifier]

    r = list(table.select('id', 'x', 'y', 'z').execute(db=db))[0]

    out = (
        table.like({'x': r['x']}, vector_index='test_vector_search', n=10)
        .select('id')
        .execute(db)
    )
    ids = list(out.scores.keys())
    scores = out.scores

    assert str(r['id']) in ids[:3]
    assert scores[ids[0]] > 0.999999

    # pre-like
    result = (
        table.like(Document({'x': r['x']}), vector_index='test_vector_search', n=1)
        .select('id')
        .execute(db)
    )

    result = list(result)
    assert result[0]['id'] == r['id']

    # post-like
    q = table.select('id').like(
        Document({'x': r['x']}), vector_index='test_vector_search', n=3
    )
    result = list(q.execute(db))
    assert len(result) == 3
    assert result[0]['id'] == r['id']


@pytest.mark.parametrize("db", [DBConfig.mongodb], indirect=True)
def test_model(db):
    import torch

    m = db.load('model', 'linear_a')

    m.predict_one(torch.randn(32))

    from superduperdb.backends.base.query import Model

    t = torch.randn(32)

    m.predict_one(t)

    q = Model('linear_a').predict_one(t)

    out = db.execute(q).unpack()

    assert isinstance(out, torch.Tensor)


def test_builder():
    q = MongoQuery(identifier='table', parts=()).select('id').where(2, 3, 4, a=5)
    assert str(q) == 'table.select("id").where(2, 3, 4, a=5)'


multi_query = """something.find().limit(5)
other_thing.join(query[0]).filter(documents[0])"""


def test_parse_and_dump():
    from superduperdb.backends.base.query import parse_query

    q = parse_query(
        documents=[], query='collection.find().limit(5)', builder_cls=MongoQuery
    )
    print('\n')
    print(q)

    q = parse_query(
        documents=[{'txt': 'test'}], query=multi_query, builder_cls=MongoQuery
    )

    print('\n')
    print(q)

    out, docs = q._dump_query()

    assert out == multi_query

    assert set(docs[0].keys()) == {'txt'}

    r = q.encode()

    import pprint

    pprint.pprint(r)


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_execute(db):
    q = db['test_coll'].insert_many([{'txt': lorem.sentence()} for _ in range(20)])

    q.execute()

    r = db['test_coll'].find_one().execute()

    assert 'txt' in r

    q = db['test_coll'].find_one()

    print(q)

    r = q.execute()


def test_serialize_with_image():
    import PIL.Image

    from superduperdb.backends.mongodb import MongoQuery
    from superduperdb.ext.pillow import pil_image

    img = PIL.Image.open('test/material/data/test.png')
    img = img.resize((2, 2))

    r = Document({'img': pil_image(img)})

    q = MongoQuery(identifier='test_coll').like(r).find()
    print(q)

    s = q.encode()

    import pprint

    pprint.pprint(s)

    decode_q = Document.decode(s).unpack()

    print(decode_q)


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_insert(db):
    table_or_collection = db['documents']
    datas = [Document({'x': i, 'y': str(i)}) for i in range(10)]
    table_or_collection.insert_many(datas).execute()

    datas_from_db = list(table_or_collection.find().execute())

    for d, d_db in zip(datas, datas_from_db):
        assert d['x'] == d_db['x']
        assert d['y'] == d_db['y']


@pytest.mark.parametrize("db", [DBConfig.sqldb_empty], indirect=True)
def test_insert_with_schema(db):
    import numpy as np
    import PIL.Image

    from superduperdb.ext.numpy.encoder import NumpyDataTypeFactory
    from superduperdb.ext.pillow.encoder import pil_image

    data = {
        'img': PIL.Image.open('test/material/data/test.png'),
        'array': np.array([1, 2, 3]),
    }

    schema = Schema(
        'schema',
        fields={
            'img': pil_image,
            'array': NumpyDataTypeFactory.create(data['array']),
        },
    )

    table = Table('documents', schema=schema)
    db.add(table)

    table_or_collection = db['documents']
    datas = [Document(data)]

    table_or_collection.insert(datas).do_execute()
    datas_from_db = list(table_or_collection.select().do_execute())

    for d, d_db in zip(datas, datas_from_db):
        assert d['img'].size == d_db['img'].size
        assert np.all(d['array'] == d_db['array'])
