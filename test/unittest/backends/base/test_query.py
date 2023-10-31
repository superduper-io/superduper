from superduperdb.base.document import Document


def test_execute_insert_and_find(data_layer, empty_collection):
    collection = empty_collection
    collection.insert_many([Document({'this': 'is a test'})]).execute(data_layer)
    r = collection.find_one().execute(data_layer)
    print(r)


def test_execute_complex_query(data_layer, empty_collection):
    collection = empty_collection
    collection.insert_many(
        [Document({'this': f'is a test {i}'}) for i in range(100)]
    ).execute(data_layer)

    cur = collection.find().limit(10).sort('this', -1).execute(data_layer)
    for r in cur:
        print(r)


def test_execute_like_queries(data_layer):
    from superduperdb.backends.mongodb.query import Collection

    collection = Collection('documents')
    # get a data point for testing
    r = collection.find_one().execute(data_layer)

    out = (
        collection.like({'x': r['x']}, vector_index='test_vector_search', n=10)
        .find()
        .execute(data_layer)
    )

    ids = list(out.scores.keys())
    scores = out.scores

    assert str(r['_id']) in ids[:3]
    assert scores[ids[0]] > 0.999999

    # pre-like
    result = (
        collection.like(Document({'x': r['x']}), vector_index='test_vector_search', n=1)
        .find_one()
        .execute(data_layer)
    )

    print(result)
    assert result['_id'] == r['_id']

    q = collection.find().like(
        Document({'x': r['x']}), vector_index='test_vector_search', n=3
    )

    print(q)

    # check queries we didn't have before
    y = collection.distinct('y').execute(data_layer)
    assert set(y) == {0, 1}

    # post-like
    result = q.execute(data_layer)
