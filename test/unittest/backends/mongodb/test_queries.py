import PIL.PngImagePlugin
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
@pytest.mark.parametrize('local_collection_with_random_data', [5], indirect=True)
def test_delete_many(data_layer, local_collection_with_random_data):
    collection = local_collection_with_random_data
    old_ids = {r['_id'] for r in data_layer.execute(collection.find({}, {'_id': 1}))}
    deleted_ids = list(old_ids)[:2]
    data_layer.execute(collection.delete_many({'_id': {'$in': deleted_ids}}))
    new_ids = {r['_id'] for r in data_layer.execute(collection.find({}, {'_id': 1}))}
    assert len(new_ids) == 3

    assert old_ids - new_ids == set(deleted_ids)


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize('local_data_layer', [5], indirect=True)
def test_replace(local_data_layer):
    collection = Collection('documents')
    r = next(local_data_layer.execute(collection.find()))
    x = torch.randn(32)
    t = local_data_layer.encoders['torch.float32[32]']
    new_x = t(x)
    r['x'] = new_x
    local_data_layer.execute(
        collection.replace_one(
            {'_id': r['_id']},
            r,
        )
    )

    new_r = local_data_layer.execute(collection.find_one({'_id': r['_id']}))
    assert new_r['x'].x.tolist() == new_x.x.tolist()


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_insert_from_uris(local_data_layer, empty_collection, image_url):
    collection = empty_collection
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
    local_data_layer.execute(collection.insert_many(to_insert))
    r = local_data_layer.execute(collection.find_one())
    assert isinstance(r['item'].x, PIL.Image.Image)
    assert isinstance(r['other']['item'].x, PIL.Image.Image)


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize('local_data_layer', [5], indirect=True)
def test_update_many(local_data_layer):
    collection = Collection('documents')
    to_update = torch.randn(32)
    t = local_data_layer.encoders['torch.float32[32]']
    local_data_layer.execute(
        collection.update_many({}, Document({'$set': {'x': t(to_update)}}))
    )
    cur = local_data_layer.execute(collection.find())
    r = next(cur)
    s = next(cur)

    assert all(r['x'].x == to_update)
    assert all(s['x'].x == to_update)
    assert (
        r['_outputs']['x']['linear_a'].x.tolist()
        == s['_outputs']['x']['linear_a'].x.tolist()
    )


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize('local_data_layer', [5], indirect=True)
def test_insert_many(local_data_layer):
    collection = Collection('documents')
    an_update = get_new_data(
        local_data_layer.encoders['torch.float32[32]'], 10, update=True
    )
    local_data_layer.execute(collection.insert_many(an_update))
    r = next(local_data_layer.execute(collection.find({'update': True})))
    assert 'linear_a' in r['_outputs']['x']
    assert len(list(local_data_layer.execute(collection.find()))) == 5 + 10


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_like(data_layer):
    collection = Collection('documents')
    r = data_layer.execute(collection.find_one())
    query = collection.like(
        r=Document({'x': r['x']}),
        vector_index='test_vector_search',
    ).find()
    s = next(data_layer.execute(query))
    assert r['_id'] == s['_id']


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize('local_data_layer', [5], indirect=True)
def test_insert_one(local_data_layer):
    # MARK: empty Collection + a_single_insert
    collection = Collection('documents')
    a_single_insert = get_new_data(
        local_data_layer.encoders['torch.float32[32]'], 1, update=False
    )[0]
    print(a_single_insert)
    out, _ = local_data_layer.execute(collection.insert_one(a_single_insert))
    r = local_data_layer.execute(collection.find({'_id': out[0]}))
    docs = list(r)
    assert docs[0]['x'].x.tolist() == a_single_insert['x'].x.tolist()


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_delete_one(data_layer, local_collection_with_random_data):
    # MARK: random data (change)
    collection = local_collection_with_random_data
    r = data_layer.execute(collection.find_one())
    data_layer.execute(collection.delete_one({'_id': r['_id']}))
    with pytest.raises(StopIteration):
        next(data_layer.execute(Collection('documents').find({'_id': r['_id']})))


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_find(data_layer):
    collection = Collection('documents')
    r = data_layer.execute(collection.find().limit(1))
    assert len(list(r)) == 1
    r = data_layer.execute(collection.find().limit(100))
    assert len(list(r)) == 100


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_find_one(data_layer):
    r = data_layer.execute(Collection('documents').find_one())
    assert isinstance(r, Document)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_aggregate(data_layer):
    r = data_layer.execute(
        Collection('documents').aggregate([{'$sample': {'size': 1}}])
    )
    assert len(list(r)) == 1


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize('local_data_layer', [5], indirect=True)
def test_replace_one(local_data_layer):
    collection = Collection('documents')
    # MARK: random data (change)
    new_x = torch.randn(32)
    t = local_data_layer.encoders['torch.float32[32]']
    r = local_data_layer.execute(collection.find_one())
    r['x'] = t(new_x)
    local_data_layer.execute(collection.replace_one({'_id': r['_id']}, r))
    doc = local_data_layer.execute(collection.find_one({'_id': r['_id']}))
    assert doc.unpack()['x'].tolist() == new_x.tolist()
