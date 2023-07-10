from superduperdb.datalayer.mongodb.query import Collection


def test_find(random_data):
    r = random_data.execute(Collection(name='documents').find_one())
    print(r['x'].x.shape)

    cursor = random_data.execute(Collection(name='documents').find())
    print(next(cursor))
