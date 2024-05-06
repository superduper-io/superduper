from superduperdb.backends.mongodb.new_query import Collection


def test_add_fold():

    coll = Collection(identifier='test_coll')

    q = coll.find().limit(5)

    new_q = q.add_fold('valid')

    print(new_q)

    assert str(new_q) == "test_coll.find({'_fold': 'valid'}).limit(5)"


def test_add():
    ...
