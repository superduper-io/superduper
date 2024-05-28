import random
from test.db_config import DBConfig

import numpy as np
import pytest

from superduperdb.backends.mongodb import query as q
from superduperdb.backends.mongodb.query import MongoQuery
from superduperdb.base.config import BytesEncoding
from superduperdb.base.document import Document
from superduperdb.components.schema import Schema
from superduperdb.ext.numpy.encoder import array


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


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_mongo_without_schema(db):
    collection_name = "documents"
    array_tensor = array(dtype="float64", shape=(32,))
    db.add(array_tensor)
    data = []

    for id_ in range(5):
        x = np.random.rand(32)
        y = int(random.random() > 0.5)
        z = np.random.randn(32)
        data.append(
            Document(
                {
                    "id": id_,
                    "x": array_tensor(x),
                    "y": y,
                    "z": array_tensor(z),
                }
            )
        )

    db.execute(
        MongoQuery(collection_name).insert_many(data),
    )
    collection = MongoQuery(collection_name)
    r = collection.find_one().do_execute(db)
    rs = list(collection.find().do_execute(db))

    rs = sorted(rs, key=lambda x: x['id'])

    assert np.array_equal(r['x'].x, data[0]['x'].x)
    assert np.array_equal(r['z'].x, data[0]['z'].x)

    assert np.array_equal(rs[0]['x'].x, data[0]['x'].x)
    assert np.array_equal(rs[0]['z'].x, data[0]['z'].x)


@pytest.mark.parametrize(
    "db,schema",
    [
        (DBConfig.mongodb_empty, BytesEncoding.BASE64),
        (DBConfig.mongodb_empty, BytesEncoding.BYTES),
    ],
    indirect=True,
)
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
        MongoQuery(db=db, identifier=collection_name).insert_many(data),
    )
    r = db[collection_name].find_one().execute()
    rs = list(db[collection_name].find().execute())

    rs = sorted(rs, key=lambda x: x['id'])

    assert np.array_equal(r['x'], gt['x'])
    assert np.array_equal(r['z'], gt['z'])

    assert np.array_equal(rs[0]['x'], gt['x'])
    assert np.array_equal(rs[0]['z'], gt['z'])


def test_select_missing_outputs(db):
    docs = list(db.execute(q.MongoQuery('documents').find({}, {'_id': 1})))
    ids = [r['_id'] for r in docs[: len(docs) // 2]]
    db.execute(
        q.MongoQuery('documents').update_many(
            {'_id': {'$in': ids}},
            Document({'$set': {'_outputs.x::test_model_output::0::0': 'test'}}),
        )
    )
    select = q.MongoQuery('documents').find({}, {'_id': 1})
    modified_select = select.select_ids_of_missing_outputs('x::test_model_output::0::0')
    out = list(db.execute(modified_select))
    assert len(out) == (len(docs) - len(ids))


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_special_query_serialization(db):
    q2 = db['docs'].find({'x': {'$lt': 9}})
    encoded_query = q2.encode()
    base = encoded_query['_base'][1:]
    assert encoded_query['_leaves'][base]['documents'][0] == {'x': {'<$>lt': 9}}

    rq2 = Document.decode(encoded_query).unpack()
    assert rq2.parts[0][1][0] == {'x': {'$lt': 9}}
