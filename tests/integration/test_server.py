import uuid

import pytest
import torch

from superduperdb import CFG
from superduperdb.core.documents import Document
from superduperdb.core.encoder import Encoder
from superduperdb.datalayer.mongodb.query import Collection
from superduperdb.encoders.torch.tensor import tensor
from superduperdb.serve.client import Client


@pytest.fixture(scope="module")
def client(test_server):  # Warning: Magic so that test_server is started, don't remove!
    return Client(CFG.server.uri)


@pytest.fixture(
    scope="function"
)  # scope="function" so that each test gets a new collection
def test_collection():
    collection_name = str(uuid.uuid4())
    return Collection(name=collection_name)


def test_load(client):
    encoder = client.load('encoder', 'torch.float32[32]')
    assert isinstance(encoder, Encoder)


def test_show(client):
    encoders = client.show('encoder')
    assert encoders == ['torch.float32[16]', 'torch.float32[32]']


def test_select_one(
    client, database_with_default_encoders_and_model, test_collection, fake_inserts
):
    database_with_default_encoders_and_model.execute(
        test_collection.insert_many([fake_inserts[0]])
    )
    r = database_with_default_encoders_and_model.execute(test_collection.find_one())
    s = client.execute(test_collection.find_one())
    assert r['_id'] == s['_id']


def test_insert(
    client, database_with_default_encoders_and_model, test_collection, fake_inserts
):
    client.execute(test_collection.insert_many([fake_inserts[0]]))
    r = database_with_default_encoders_and_model.execute(test_collection.find_one())
    assert all(torch.eq(r['x'].x, fake_inserts[0]['x'].x))


def test_remove(client, database_with_default_encoders_and_model):
    database_with_default_encoders_and_model.add(tensor(torch.float64, shape=(32,)))
    encoders = client.show('encoder')
    assert encoders == ['torch.float32[16]', 'torch.float32[32]', 'torch.float64[32]']

    client.remove('encoder', 'torch.float64[32]', force=True)
    encoders = client.show('encoder')
    assert encoders == ['torch.float32[16]', 'torch.float32[32]']


def test_update(
    client, database_with_default_encoders_and_model, test_collection, fake_inserts
):
    database_with_default_encoders_and_model.execute(
        test_collection.insert_many([fake_inserts[0]])
    )
    encoder = database_with_default_encoders_and_model.encoders['torch.float32[32]']
    updated_values = torch.randn(32)
    client.execute(
        test_collection.update_many(
            {}, Document({'$set': {'x': encoder(updated_values)}})
        )
    )
    r = database_with_default_encoders_and_model.execute(test_collection.find_one())
    assert all(torch.eq(r['x'].x, updated_values))
