from superduperdb.backends.ibis.field_types import dtype
from superduperdb.component.document import Document
from superduperdb.component.schema import Schema
from superduperdb.ext.pillow.image import pil_image


def test_create_ibis_query():
    from superduperdb.backends.ibis.query import IbisTable

    t = IbisTable(
        identifier='my_table',
        schema=Schema(
            identifier='my_table',
            fields={
                'id': dtype('int64'),
                'health': dtype('int32'),
                'age': dtype('int32'),
                'image': pil_image,
            },
        ),
    )

    q = t.filter(t.age > 0).select('age')

    print(q)

    q = t.like({'this': 'is a test'}).filter(t.age > 0).select('age')

    print(q)

    print(q.pre_like)
    print(q.query_linker)


def test_execute_insert_and_find(empty):
    from superduperdb.backends.mongodb.query import Collection
    from superduperdb.component.document import Document

    collection = Collection('documents')
    collection.insert_many([Document({'this': 'is a test'})]).execute(empty)
    r = collection.find_one().execute(empty)
    print(r)


def test_execute_complex_query(empty):
    from superduperdb.backends.mongodb.query import Collection
    from superduperdb.component.document import Document

    collection = Collection('documents')
    collection.insert_many(
        [Document({'this': f'is a test {i}'}) for i in range(100)]
    ).execute(empty)

    cur = collection.find().limit(10).sort('this', -1).execute(empty)
    for r in cur:
        print(r)


def test_execute_like_queries(with_vector_index):
    from superduperdb.backends.mongodb.query import Collection

    collection = Collection('documents')

    # get a data point for testing
    r = collection.find_one().execute(with_vector_index)

    out = (
        collection.like({'x': r['x']}, vector_index='test_vector_search', n=10)
        .find()
        .execute(with_vector_index)
    )

    ids = list(out.scores.keys())
    scores = out.scores

    assert str(r['_id']) in ids[:3]
    assert scores[ids[0]] > 0.999999

    # pre-like
    result = (
        collection.like(Document({'x': r['x']}), vector_index='test_vector_search', n=1)
        .find_one()
        .execute(with_vector_index)
    )

    print(result)
    assert result['_id'] == r['_id']

    q = collection.find().like(
        Document({'x': r['x']}), vector_index='test_vector_search', n=3
    )

    print(q)

    # check queries we didn't have before
    y = collection.distinct('y').execute(with_vector_index)
    assert set(y) == {0, 1}

    # post-like
    result = q.execute(with_vector_index)
