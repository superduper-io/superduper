import lorem
import pytest

from superduper.backends.ibis.field_types import dtype
from superduper.backends.mongodb.query import MongoQuery
from superduper.base.document import Document
from superduper.components.schema import Schema
from superduper.components.table import Table


@pytest.fixture
def table():
    schema = Schema('schema', fields={'id': dtype('str'), 'this': dtype('str')})
    table = Table('documents', schema=schema)
    return table


def test_execute_insert_and_find(db, table):
    db.add(table)
    t = db['documents']
    insert_query = t.insert([Document({'this': 'is a test'})])
    db.execute(insert_query)
    r = list(t.select().execute())[0]
    assert r['this'] == 'is a test'


def test_execute_like_queries(db):
    from test.utils.setup.fake_data import add_models, add_random_data, add_vector_index

    add_random_data(db)
    add_models(db)
    add_vector_index(db)
    table = db['documents']
    # Get a data point for testing
    r = list(table.select().execute())[0]

    out = (
        table.like({'x': r['x']}, vector_index='test_vector_search', n=10)
        .select()
        .do_execute(db)
    )
    primary_id = table.primary_id
    scores = out.scores

    ids = [o[primary_id] for o in list(out)]

    assert r[primary_id] in ids
    assert scores[str(ids[0])] > 0.999999

    # Pre-like
    result = (
        table.like(Document({'x': r['x']}), vector_index='test_vector_search', n=1)
        .select()
        .execute()
    )

    result = list(result)[0]

    assert result[primary_id] == r[primary_id]

    # Post-like
    q = table.select().like(
        Document({'x': r['x']}), vector_index='test_vector_search', n=3
    )
    result = q.do_execute(db)
    result = list(result)
    assert len(result) == 3
    assert result[0][primary_id] == r[primary_id]


def test_model(db):
    from test.utils.setup.fake_data import add_models

    add_models(db)
    import torch

    m = db.load('model', 'linear_a')

    m.predict(torch.randn(32))

    from superduper.backends.base.query import Model

    t = torch.randn(32)

    m.predict(t)

    q = Model(table='linear_a').predict(t)

    out = db.execute(q).unpack()

    assert isinstance(out, torch.Tensor)


def test_builder():
    q = MongoQuery(table='table', parts=()).find('id').where(2, 3, 4, a=5)
    assert str(q) == 'table.find("id").where(2, 3, 4, a=5)'


multi_query = """something.find().limit(5)
other_thing.join(query[0]).find_one(documents[0])"""


def test_parse_and_dump():
    from superduper.backends.base.query import parse_query

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


def test_execute(db):
    db.cfg.auto_schema = True
    q = db['test_coll'].insert([{'txt': lorem.sentence()} for _ in range(20)])

    q.execute()

    r = list(db['test_coll'].select().execute())[0]

    assert 'txt' in r


def test_serialize_with_image():
    import PIL.Image

    from superduper.backends.mongodb import MongoQuery
    from superduper.ext.pillow import pil_image

    img = PIL.Image.open('test/material/data/test.png')
    img = img.resize((2, 2))

    r = Document({'img': pil_image(img)})

    q = MongoQuery(table='test_coll').like(r).find()
    print(q)

    s = q.encode()

    import pprint

    pprint.pprint(s)

    decode_q = Document.decode(s).unpack()

    print(decode_q)


def test_insert(db):
    db.cfg.auto_schema = True
    table_or_collection = db['documents']
    datas = [Document({'x': i, 'y': str(i)}) for i in range(10)]
    table_or_collection.insert(datas).execute()

    datas_from_db = list(table_or_collection.select().execute())

    for d, d_db in zip(datas, datas_from_db):
        assert d['x'] == d_db['x']
        assert d['y'] == d_db['y']


def test_insert_with_schema(db):
    db.cfg.auto_schema = True
    import numpy as np
    import PIL.Image

    data = {
        'img': PIL.Image.open('test/material/data/test.png'),
        'array': np.array([1, 2, 3]),
    }

    table_or_collection = db['documents']
    datas = [Document(data)]

    table_or_collection.insert(datas).execute()
    # Make sure multiple insert works
    table_or_collection.insert(datas).execute()
    datas_from_db = list(table_or_collection.select().execute())

    for d, d_db in zip(datas, datas_from_db):
        assert d['img'].size == d_db['img'].size
        assert np.all(d['array'] == d_db['array'])


def test_insert_with_diff_schemas(db):
    db.cfg.auto_schema = True
    import numpy as np
    import PIL.Image

    table_or_collection = db['documents']
    data = {
        'array': np.array([1, 2, 3]),
    }
    datas = [Document(data)]
    table_or_collection.insert(datas).execute()

    datas_from_db = list(table_or_collection.select().execute())

    assert np.all(datas[0]['array'] == datas_from_db[0]['array'])

    data = {
        'img': PIL.Image.open('test/material/data/test.png'),
    }
    datas = [Document(data)]

    # Do not support different schema
    with pytest.raises(Exception):
        table_or_collection.insert(datas).execute()


def test_auto_document_wrapping(db):
    db.cfg.auto_schema = True
    import numpy as np

    table_or_collection = db['my_table']
    data = {'x': np.zeros((1))}
    datas = [Document(data)]
    table_or_collection.insert(datas).execute()

    def _check(n):
        c = list(table_or_collection.select().execute())
        assert len(c) == n
        return c

    _check(1)

    # Without `Document` dict data
    table_or_collection.insert([data]).execute()
    _check(2)

    # Without `Document` non dict data
    table_or_collection.insert([np.zeros((1))]).execute()
    c = _check(3)

    gt = np.zeros((1))

    # Auto wrapped _base
    assert 'x' in c[-1]
    assert c[-1].unpack()['x'] == gt


def test_model_query():
    from superduper.backends.base.query import Model

    q = Model(table='my-model').predict('This is a test')

    r = q.dict()
    assert r['query'] == 'my-model.predict("This is a test")'
