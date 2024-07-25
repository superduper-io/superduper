import random

import numpy as np
import pytest
from superduper.base.document import Document
from superduper.components.schema import Schema
from superduper.ext.numpy.encoder import array

from superduper_mongodb.query import MongoQuery


@pytest.fixture
def schema(request):
    bytes_encoding = request.param if hasattr(request, 'param') else None

    array_tensor = array(dtype="float64", shape=(32,), bytes_encoding=bytes_encoding)
    schema = Schema(
        identifier=f'documents-{bytes_encoding}',
        fields={
            "x": array_tensor,
            "z": array_tensor,
        },
    )
    return schema


def test_mongo_schema(db, schema):
    collection_name = "documents"
    data = []

    for id_ in range(5):
        x = np.random.rand(32)
        y = int(random.random() > 0.5)
        z = np.random.randn(32)
        data.append(
            Document(
                {
                    "id": id_,
                    "x": x,
                    "y": y,
                    "z": z,
                },
                db=db,
                schema=schema,
            )
        )

    db.add(schema)
    gt = data[0]

    db.execute(
        MongoQuery(db=db, table=collection_name).insert_many(data),
    )
    r = db[collection_name].find_one().execute()
    rs = list(db[collection_name].find().execute())

    rs = sorted(rs, key=lambda x: x['id'])

    assert np.array_equal(r['x'], gt['x'])
    assert np.array_equal(r['z'], gt['z'])

    assert np.array_equal(rs[0]['x'], gt['x'])
    assert np.array_equal(rs[0]['z'], gt['z'])


def test_select_missing_outputs(db):
    docs = list(db.execute(MongoQuery(table='documents').find({}, {'_id': 1})))
    ids = [r['_id'] for r in docs[: len(docs) // 2]]
    db.execute(
        MongoQuery(table='documents').update_many(
            {'_id': {'$in': ids}},
            Document({'$set': {'_outputs__x::test_model_output::0::0': 'test'}}),
        )
    )
    select = MongoQuery(table='documents').find({}, {'_id': 1})
    modified_select = select.select_ids_of_missing_outputs('x::test_model_output::0::0')
    out = list(db.execute(modified_select))
    assert len(out) == (len(docs) - len(ids))


def test_special_query_serialization(db):
    q2 = db['docs'].find({'x': {'$lt': 9}})
    encoded_query = q2.encode()
    base = encoded_query['_base'][1:]
    assert encoded_query['_builds'][base]['documents'][0] == {'x': {'<$>lt': 9}}

    rq2 = Document.decode(encoded_query).unpack()
    assert rq2.parts[0][1][0] == {'x': {'$lt': 9}}


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
