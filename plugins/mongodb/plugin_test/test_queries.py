import random
from test.utils.setup.fake_data import (
    add_listeners,
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


def test_outputs_query_2(db):
    import numpy
    from superduper import model

    db.cfg.auto_schema = True

    @model
    def test(x):
        return numpy.random.randn(32)

    db['example'].insert([{'x': f'test {i}', 'y': f'other {i}'} for i in range(5)]).execute()

    l1 = test.to_listener(key='x', select=db['example'].select(), identifier='l-x')
    l2 = test.to_listener(key='y', select=db['example'].select(), identifier='l-y')

    db.apply(l1)
    db.apply(l2)

    @model(flatten=True)
    def test_flat(x):
        return [numpy.random.randn(32) for _ in range(3)]

    l3 = test_flat.to_listener(key='x', select=db['example'].select(), identifier='l-x-flat')

    db.apply(l3)    

    ########

    q = db['example'].outputs(l1.predict_id)

    results = q.execute().tolist()

    assert len(results) == 5

    #######

    q = db['example'].outputs(l2.predict_id)

    results = q.execute().tolist()

    assert len(results) == 5

    #######

    q = db['example'].outputs(l1.predict_id, l2.predict_id)

    results = q.execute().tolist()

    assert len(results) == 5

    #######

    q = db['example'].outputs(l3.predict_id)

    results = q.execute().tolist()

    assert len(results) == 15

    #######

    q = db['example'].outputs(l1.predict_id, l3.predict_id)

    results = q.execute().tolist()

    assert len(results) == 15

    #######

    q = db['example'].outputs(l1.predict_id, l2.predict_id, l3.predict_id)

    results = q.execute().tolist()

    assert len(results) == 15


def test_outputs_query(db):
    db.cfg.auto_schema = True

    add_random_data(db, n=5)
    add_models(db)

    import numpy

    l1, l2, l1_flat = add_listeners(db)
    
    sample1 = db[l1.outputs].find_one().execute()
    sample2 = db[l2.outputs].find_one().execute()

    outputs_1 = list(db['documents'].outputs(l1.predict_id).execute())
    assert len(outputs_1) == 5
    outputs_2 = list(db['documents'].outputs(l2.predict_id).execute())
    assert len(outputs_2) == 5
    outputs_1_2 = next(db['documents'].outputs(l1.predict_id, l2.predict_id).execute())
    assert len(outputs_1_2) == 5

    # outputs_1_flat = list(db['documents'].outputs(l1_flat.predict_id).execute())
    # # TODO this output seems to be far too big
    # # the question is what is the expected output when we have multiple listeners?
    # outputs_1_flat_2 = next(db['documents'].outputs(l1_flat.predict_id, l2.predict_id).execute())

    # assert isinstance(outputs_1[l1.outputs], np.ndarray)
    # assert isinstance(outputs_2[l2.outputs], np.ndarray)

    # assert isinstance(outputs_1_2[l1.outputs], np.ndarray)
    # assert isinstance(outputs_1_2[l2.outputs], np.ndarray)


def test_insert_many(db):
    add_random_data(db, n=5)
    add_models(db)
    add_listeners(db)
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
