from superduperdb.db.mongodb.query import Collection


def test_find(database_with_random_tensor_data):
    r = database_with_random_tensor_data.execute(
        Collection(name='documents').find_one()
    )
    print(r['x'].x.shape)

    cursor = database_with_random_tensor_data.execute(
        Collection(name='documents').find()
    )
    print(next(cursor))
