import uuid

import pytest
import torch

from superduperdb import CFG
from superduperdb.container.document import Document
from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.torch.model import TorchModel
from superduperdb.ext.torch.tensor import tensor
from superduperdb.server.client import Client


@pytest.fixture(scope="function")
def client(test_server):  # Warning: Magic so that test_server is started, don't remove!
    return Client(CFG.server.uri)


@pytest.fixture(
    scope="function"
)  # scope="function" so that each test gets a new collection
def empty_collection():
    collection_name = str(uuid.uuid4())
    return Collection(name=collection_name)


def test_show(client):
    encoders = client.show('encoder')
    assert encoders == ['torch.float32[16]', 'torch.float32[32]']


def test_select_one(client, session_database, empty_collection, fake_inserts):
    session_database.execute(empty_collection.insert_many([fake_inserts[0]]))
    r = session_database.execute(empty_collection.find_one())
    s = client.execute(empty_collection.find_one())
    assert r['_id'] == s['_id']


def test_add_load(client, session_database):
    m = TorchModel(
        identifier='test-add-client',
        object=torch.nn.Linear(10, 20),
        encoder=tensor(torch.float, shape=(20,)),
    )
    client.add(m)
    models = session_database.show('model')
    assert 'test-add-client' in models

    m = client.load('model', 'test-add-client')
    assert isinstance(m.object.artifact, torch.nn.Module)

    session_database.remove('model', 'test-add-client', force=True)
    session_database.remove('encoder', 'torch.float32[20]', force=True)


def test_insert(client, session_database, empty_collection, fake_inserts):
    client.execute(empty_collection.insert_many([fake_inserts[0]]))
    r = session_database.execute(empty_collection.find_one())
    assert all(torch.eq(r['x'].x, fake_inserts[0]['x'].x))


def test_remove(client, session_database):
    session_database.add(tensor(torch.float64, shape=(32,)))
    encoders = client.show('encoder')
    assert encoders == ['torch.float32[16]', 'torch.float32[32]', 'torch.float64[32]']

    client.remove('encoder', 'torch.float64[32]', force=True)
    encoders = client.show('encoder')
    assert encoders == ['torch.float32[16]', 'torch.float32[32]']


def test_update(client, session_database, empty_collection, fake_inserts):
    session_database.execute(empty_collection.insert_many([fake_inserts[0]]))
    encoder = session_database.encoders['torch.float32[32]']
    updated_values = torch.randn(32)
    client.execute(
        empty_collection.update_many(
            {}, Document({'$set': {'x': encoder(updated_values)}})
        )
    )
    r = session_database.execute(empty_collection.find_one())
    assert all(torch.eq(r['x'].x, updated_values))
