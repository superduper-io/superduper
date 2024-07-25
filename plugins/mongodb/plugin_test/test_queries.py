import random
from test.utils.setup.fake_data import (
    add_listener,
    add_models,
    add_random_data,
    add_vector_index,
)

import numpy as np
import pytest
from superduper.base.document import Document


def get_new_data(n=10, update=False):
    data = []
    for _ in range(n):
        x = np.random.rand(32)
        y = int(random.random() > 0.5)
        z = np.random.rand(32)
        data.append(
            Document(
                {
                    'x': x,
                    'y': y,
                    'z': z,
                    'update': update,
                }
            )
        )
    return data


def test_delete_many(db):
    add_random_data(db, n=5)
    collection = db['documents']
    old_ids = {r['_id'] for r in db.execute(collection.find({}, {'_id': 1}))}
    deleted_ids = list(old_ids)[:2]
    db.execute(collection.delete_many({'_id': {'$in': deleted_ids}}))
    new_ids = {r['_id'] for r in db.execute(collection.find({}, {'_id': 1}))}
    assert len(new_ids) == 3

    assert old_ids - new_ids == set(deleted_ids)


def test_replace(db):
    add_random_data(db, n=5)
    collection = db['documents']
    r = next(db.execute(collection.find()))
    new_x = np.random.rand(32)
    r['x'] = new_x
    db.execute(
        collection.replace_one(
            {'_id': r['_id']},
            r,
        )
    )

    new_r = db.execute(collection.find_one({'_id': r['_id']}))
    assert new_r['x'].tolist() == new_x.tolist()


@pytest.mark.skipif(True, reason='URI not working')
def test_insert_from_uris(db, image_url):
    import PIL
    from superduper.ext.pillow.encoder import pil_image

    if image_url.startswith('file://'):
        image_url = image_url[7:]

    db.add(pil_image)
    collection = db['documents']
    to_insert = [Document({'img': pil_image(uri=image_url)})]

    db.execute(collection.insert_many(to_insert))

    r = db.execute(collection.find_one())
    assert isinstance(r['img'].x, PIL.Image.Image)


def test_update_many(db):
    add_random_data(db, n=5)
    collection = db['documents']
    to_update = np.random.randn(32)
    db.execute(collection.update_many({}, Document({'$set': {'x': to_update}})))
    cur = db.execute(collection.find())
    r = next(cur)

    assert all(r['x'] == to_update)

    # TODO: Need to support Update result in predict_in_db
    # listener = db.load('listener', 'vector-x')
    # assert all(
    #     listener.model.predict(to_update)
    #     == next(db['_outputs__vector-x'].find().execute())['_outputs__vector-x'].x
    # )


def test_insert_many(db):
    add_random_data(db, n=5)
    add_models(db)
    add_listener(db)
    collection = db['documents']
    an_update = get_new_data(10, update=True)
    db.execute(collection.insert(an_update))

    assert len(list(db.execute(collection.find()))) == 5 + 10
    assert len(list(db.execute(db['_outputs__vector-x'].find()))) == 5 + 10
    assert len(list(db.execute(db['_outputs__vector-y'].find()))) == 5 + 10


def test_like(db):
    add_random_data(db, n=5)
    add_models(db)
    add_vector_index(db)
    collection = db['documents']
    r = db.execute(collection.find_one())
    query = collection.like(
        r=Document({'x': r['x']}),
        vector_index='test_vector_search',
    ).find()
    s = next(db.execute(query))
    assert r['_id'] == s['_id']


def test_insert_one(db):
    add_random_data(db, n=5)
    # MARK: empty Collection + a_single_insert
    collection = db['documents']
    a_single_insert = get_new_data(1, update=False)[0]
    q = collection.insert_one(a_single_insert)
    out, _ = db.execute(q)
    r = db.execute(collection.find({'_id': out[0]}))
    docs = list(r)
    assert docs[0]['x'].tolist() == a_single_insert['x'].tolist()


def test_delete_one(db):
    add_random_data(db, n=5)
    collection = db['documents']
    r = db.execute(collection.find_one())
    db.execute(collection.delete_one({'_id': r['_id']}))
    with pytest.raises(StopIteration):
        next(db.execute(db['documents'].find({'_id': r['_id']})))


def test_find(db):
    add_random_data(db, n=10)
    collection = db['documents']
    r = db.execute(collection.find().limit(1))
    assert len(list(r)) == 1
    r = db.execute(collection.find().limit(5))
    assert len(list(r)) == 5


def test_find_one(db):
    add_random_data(db, n=5)
    r = db.execute(db['documents'].find_one())
    assert isinstance(r, Document)


def test_replace_one(db):
    add_random_data(db, n=5)
    collection = db['documents']
    # MARK: random data (change)
    new_x = np.random.randn(32)
    r = db.execute(collection.find_one())
    r['x'] = new_x
    db.execute(collection.replace_one({'_id': r['_id']}, r))
    doc = db.execute(collection.find_one({'_id': r['_id']}))
    print(doc['x'])
    assert doc.unpack()['x'].tolist() == new_x.tolist()
