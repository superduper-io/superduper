import pytest

try:
    import torch
except ImportError:
    torch = None

import random
from test.db_config import DBConfig

from superduperdb.backends.mongodb.query import MongoQuery
from superduperdb.base.document import Document
from superduperdb.components.datatype import DataType


def get_new_data(encoder: DataType, n=10, update=False):
    data = []
    for _ in range(n):
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
def test_delete_many(db):
    collection = MongoQuery(table='documents')
    old_ids = {r['_id'] for r in db.execute(collection.find({}, {'_id': 1}))}
    deleted_ids = list(old_ids)[:2]
    db.execute(collection.delete_many({'_id': {'$in': deleted_ids}}))
    new_ids = {r['_id'] for r in db.execute(collection.find({}, {'_id': 1}))}
    assert len(new_ids) == 3

    assert old_ids - new_ids == set(deleted_ids)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_replace(db):
    collection = MongoQuery(table='documents')
    r = next(db.execute(collection.find()))
    x = torch.randn(32)
    t = db.datatypes['torch-float32[32]']
    new_x = t(x)
    r['x'] = new_x
    db.execute(
        collection.replace_one(
            {'_id': r['_id']},
            r,
        )
    )

    new_r = db.execute(collection.find_one({'_id': r['_id']}))
    assert new_r['x'].x.tolist() == new_x.x.tolist()


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize('db', [DBConfig.mongodb_empty], indirect=True)
def test_insert_from_uris(db, image_url):
    import PIL

    from superduperdb.ext.pillow.encoder import pil_image

    if image_url.startswith('file://'):
        image_url = image_url[7:]

    db.add(pil_image)
    collection = MongoQuery(table='documents')
    to_insert = [Document({'img': pil_image(uri=image_url)})]

    db.execute(collection.insert_many(to_insert))

    r = db.execute(collection.find_one())
    assert isinstance(r['img'].x, PIL.Image.Image)


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize('db', [DBConfig.mongodb_empty], indirect=True)
def test_insert_from_uris_bytes_encoding(db, image_url):
    import PIL

    from superduperdb.base.config import BytesEncoding
    from superduperdb.ext.pillow.encoder import pil_image

    my_pil_image = DataType(
        'my_pil_image',
        encoder=pil_image.encoder,
        decoder=pil_image.decoder,
        bytes_encoding=BytesEncoding.BASE64,
    )

    db.add(my_pil_image)

    if image_url.startswith('file://'):
        image_url = image_url[7:]

    collection = MongoQuery(table='documents')
    to_insert = [Document({'img': my_pil_image(PIL.Image.open(image_url))})]

    db.execute(collection.insert_many(to_insert))

    r = db.execute(collection.find_one())
    assert isinstance(r['img'].x, PIL.Image.Image)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_update_many(db):
    collection = MongoQuery(table='documents')
    to_update = torch.randn(32)
    t = db.datatypes['torch-float32[32]']
    db.execute(collection.update_many({}, Document({'$set': {'x': t(to_update)}})))
    cur = db.execute(collection.find())
    r = next(cur)
    s = next(cur)

    assert all(r['x'].x == to_update)
    assert all(s['x'].x == to_update)
    listeners = [db.load('listener', k) for k in db.show('listener')]
    key = [k.uuid for k in listeners if k.key == 'x'][0]

    assert r['_outputs'][key].x.tolist() == s['_outputs'][key].x.tolist()


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_insert_many(db):
    collection = MongoQuery(table='documents')
    an_update = get_new_data(db.datatypes['torch-float32[32]'], 10, update=True)
    db.execute(collection.insert_many(an_update))
    r = next(db.execute(collection.find({'update': True})))

    keys = [k.split('/')[-1] for k in db.show('listener')]
    assert any([k in r['_outputs'] for k in keys])
    assert len(list(db.execute(collection.find()))) == 5 + 10


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_like(db):
    collection = MongoQuery(table='documents')
    r = db.execute(collection.find_one())
    query = collection.like(
        r=Document({'x': r['x']}),
        vector_index='test_vector_search',
    ).find()
    s = next(db.execute(query))
    assert r['_id'] == s['_id']


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_insert_one(db):
    # MARK: empty Collection + a_single_insert
    collection = MongoQuery(table='documents')
    a_single_insert = get_new_data(db.datatypes['torch-float32[32]'], 1, update=False)[
        0
    ]
    q = collection.insert_one(a_single_insert)
    out, _ = db.execute(q)
    r = db.execute(collection.find({'_id': out[0]}))
    docs = list(r)
    assert docs[0]['x'].x.tolist() == a_single_insert['x'].x.tolist()


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_delete_one(db):
    # MARK: random data (change)
    collection = MongoQuery(table='documents')
    r = db.execute(collection.find_one())
    db.execute(collection.delete_one({'_id': r['_id']}))
    with pytest.raises(StopIteration):
        next(db.execute(MongoQuery(table='documents').find({'_id': r['_id']})))


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_find(db):
    collection = MongoQuery(table='documents')
    r = db.execute(collection.find().limit(1))
    assert len(list(r)) == 1
    r = db.execute(collection.find().limit(5))
    assert len(list(r)) == 5


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_find_one(db):
    r = db.execute(MongoQuery(table='documents').find_one())
    assert isinstance(r, Document)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_replace_one(db):
    collection = MongoQuery(table='documents')
    # MARK: random data (change)
    new_x = torch.randn(32)
    t = db.datatypes['torch-float32[32]']
    r = db.execute(collection.find_one())
    r['x'] = t(new_x)
    db.execute(collection.replace_one({'_id': r['_id']}, r))
    doc = db.execute(collection.find_one({'_id': r['_id']}))
    print(doc['x'].x)
    assert doc.unpack()['x'].tolist() == new_x.tolist()
