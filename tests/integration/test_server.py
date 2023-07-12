import torch

from superduperdb.core.documents import Document
from superduperdb.serve.client import Client
from superduperdb import CFG
from superduperdb.datalayer.mongodb.query import Collection


coll = Collection(name='documents')


class Test:
    def test_load(self, random_data):
        c = Client(CFG.server.uri)
        e = c.load('encoder', 'torch.float32[32]')
        print(e)

    def test_select_one(self, random_data):
        c = Client(CFG.server.uri)
        r = random_data.execute(Collection(name='documents').find_one())
        s = c.execute(coll.find_one())
        print(r)
        print(s)

    def test_insert(self, random_data, an_update):
        c = Client(CFG.server.uri)
        n1 = random_data.db.documents.count_documents({})
        c.execute(coll.insert_many(an_update))
        n2 = random_data.db.documents.count_documents({})
        assert n2 == n1 + len(an_update)

    def test_show(self, random_data):
        c = Client(CFG.server.uri)
        encoders = c.show('encoder')
        assert encoders == ['torch.float32[32]']

    def test_update(self, random_data):
        c = Client(CFG.server.uri)
        t = c.encoders['torch.float32[32]']
        update = Document({'x': t(torch.randn(32))})
        c.execute(coll.update_many({}, update))

    def test_remove(self, random_data):
        c = Client(CFG.server.uri)
        c.remove('encoder', 'torch.float32[32]', force=True)
        encoders = c.show('encoder')
        assert encoders == []



