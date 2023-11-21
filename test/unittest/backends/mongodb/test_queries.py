import pytest

try:
    import torch
except ImportError:
    torch = None

import random

from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.document import Document
from superduperdb.components.encoder import Encoder


def get_new_data(encoder: Encoder, n=10, update=False):
    data = []
    for i in range(n):
        x = torch.randn(32)
        y = int(random.random() > 0.5)
        z = torch.randn(32)
        data.append(
            Document(
                {
                    'x': encoder(x),
                    'y': y,
                    'z': encoder(z),
                    'update': update,
                }
            )
        )
    return data


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize('local_db', [{'n_data': 5}], indirect=True)
def test_delete_many(local_db):
    collection = Collection('documents')
    old_ids = {r['_id'] for r in local_db.execute(collection.find({}, {'_id': 1}))}
    deleted_ids = list(old_ids)[:2]
    local_db.execute(collection.delete_many({'_id': {'$in': deleted_ids}}))
    new_ids = {r['_id'] for r in local_db.execute(collection.find({}, {'_id': 1}))}
    assert len(new_ids) == 3

    assert old_ids - new_ids == set(deleted_ids)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_replace(local_db):
    collection = Collection('documents')
    r = next(local_db.execute(collection.find()))
    x = torch.randn(32)
    t = local_db.encoders['torch.float32[32]']
    new_x = t(x)
    r['x'] = new_x
    local_db.execute(
        collection.replace_one(
            {'_id': r['_id']},
            r,
        )
    )

    new_r = local_db.execute(collection.find_one({'_id': r['_id']}))
    assert new_r['x'].x.tolist() == new_x.x.tolist()


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_insert_from_uris(local_empty_db, image_url):
    from superduperdb.ext.pillow.encoder import pil_image

    local_empty_db.add(pil_image)
    collection = Collection('documents')
    to_insert = [
        Document(
            {
                'item': {
                    '_content': {
                        'uri': image_url,
                        'encoder': 'pil_image',
                    }
                },
                'other': {
                    'item': {
                        '_content': {
                            'uri': image_url,
                            'encoder': 'pil_image',
                        }
                    }
                },
            }
        )
        for _ in range(2)
    ]
    local_empty_db.execute(collection.insert_many(to_insert))
    '''

    r = local_empty_db.execute(collection.find_one())
    assert isinstance(r['item'].x, PIL.Image.Image)
    assert isinstance(r['other']['item'].x, PIL.Image.Image)
    '''


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_update_many(local_db):
    collection = Collection('documents')
    to_update = torch.randn(32)
    t = local_db.encoders['torch.float32[32]']
    local_db.execute(
        collection.update_many({}, Document({'$set': {'x': t(to_update)}}))
    )
    cur = local_db.execute(collection.find())
    r = next(cur)
    s = next(cur)

    assert all(r['x'].x == to_update)
    assert all(s['x'].x == to_update)
    assert (
        r['_outputs']['x']['linear_a']['0'].x.tolist()
        == s['_outputs']['x']['linear_a']['0'].x.tolist()
    )


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize('local_db', [{'n_data': 5}], indirect=True)
def test_insert_many(local_db):
    collection = Collection('documents')
    an_update = get_new_data(local_db.encoders['torch.float32[32]'], 10, update=True)
    local_db.execute(collection.insert_many(an_update))
    r = next(local_db.execute(collection.find({'update': True})))
    assert 'linear_a' in r['_outputs']['x']
    assert len(list(local_db.execute(collection.find()))) == 5 + 10


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_like(db):
    collection = Collection('documents')
    r = db.execute(collection.find_one())
    query = collection.like(
        r=Document({'x': r['x']}),
        vector_index='test_vector_search',
    ).find()
    s = next(db.execute(query))
    assert r['_id'] == s['_id']


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_insert_one(local_db):
    # MARK: empty Collection + a_single_insert
    collection = Collection('documents')
    a_single_insert = get_new_data(
        local_db.encoders['torch.float32[32]'], 1, update=False
    )[0]
    out, _ = local_db.execute(collection.insert_one(a_single_insert))
    r = local_db.execute(collection.find({'_id': out[0]}))
    docs = list(r)
    assert docs[0]['x'].x.tolist() == a_single_insert['x'].x.tolist()


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_delete_one(local_db):
    # MARK: random data (change)
    collection = Collection('documents')
    r = local_db.execute(collection.find_one())
    local_db.execute(collection.delete_one({'_id': r['_id']}))
    with pytest.raises(StopIteration):
        next(local_db.execute(Collection('documents').find({'_id': r['_id']})))


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_find(db):
    collection = Collection('documents')
    r = db.execute(collection.find().limit(1))
    assert len(list(r)) == 1
    r = db.execute(collection.find().limit(100))
    assert len(list(r)) == 100


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_find_one(db):
    r = db.execute(Collection('documents').find_one())
    assert isinstance(r, Document)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_aggregate(db):
    r = db.execute(Collection('documents').aggregate([{'$sample': {'size': 1}}]))
    assert len(list(r)) == 1


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize('local_db', [{'n_data': 5}], indirect=True)
def test_replace_one(local_db):
    collection = Collection('documents')
    # MARK: random data (change)
    new_x = torch.randn(32)
    t = local_db.encoders['torch.float32[32]']
    r = local_db.execute(collection.find_one())
    r['x'] = t(new_x)
    local_db.execute(collection.replace_one({'_id': r['_id']}, r))
    doc = local_db.execute(collection.find_one({'_id': r['_id']}))
    assert doc.unpack()['x'].tolist() == new_x.tolist()
