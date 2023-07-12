import torch

from superduperdb.core.documents import Document
from superduperdb.serve.client import Client
from superduperdb import CFG
from superduperdb.datalayer.mongodb.query import Collection

from superduperdb.core.encoder import Encoder


coll = Collection(name='documents')


def test_load(random_data):
    c = Client(CFG.server.uri)
    e = c.load('encoder', 'torch.float32[32]')
    print(e)
    assert isinstance(e, Encoder)


def test_select_one(random_data):
    c = Client(CFG.server.uri)
    r = random_data.execute(Collection(name='documents').find_one())
    s = c.execute(coll.find_one())
    assert r['_id'] == s['_id']
    print(r)
    print(s)


def test_insert(random_data, an_update):
    c = Client(CFG.server.uri)
    n1 = random_data.db.documents.count_documents({})
    c.execute(coll.insert_many(an_update))
    n2 = random_data.db.documents.count_documents({})
    assert n2 == n1 + len(an_update)


def test_show(random_data):
    c = Client(CFG.server.uri)
    encoders = c.show('encoder')
    assert encoders == ['torch.float32[32]']


def test_update(random_data):
    c = Client(CFG.server.uri)
    t = c.encoders['torch.float32[32]']
    update = Document({'$set': {'x': t(torch.randn(32))}})
    c.execute(coll.update_many({}, update))
    all_x_0 = [
        r['x'].x.tolist()[0]
        for r in random_data.execute(coll.find({}, {'_id': 0, 'x': 1}))
    ]
    assert len(set(all_x_0)) == 1


def test_remove(random_data):
    c = Client(CFG.server.uri)
    c.remove('encoder', 'torch.float32[32]', force=True)
    encoders = c.show('encoder')
    assert encoders == []
