import uuid

import pytest

try:
    import torch

    from superduperdb.ext.torch.model import TorchModel
    from superduperdb.ext.torch.tensor import tensor
except ImportError:
    torch = None

from superduperdb import CFG
from superduperdb.base.document import Document
from superduperdb.db.mongodb.query import Collection
from superduperdb.server.client import Client
from superduperdb.server.server import make_flask_app


class Delegate:
    def __init__(self, delegate):
        self._delegate = delegate

    def __getattr__(self, k):
        return getattr(self._delegate, k)


class TestRequest(Delegate):
    def get(self, *a, params=None, **ka):
        if params is not None:
            ka['query_string'] = params
        response = self._delegate.get(*a, **ka)
        return TestResponse(response)


class TestResponse(Delegate):
    def json(self):
        return self._delegate.json

    @property
    def content(self):
        return self.data


@pytest.fixture
def client(database_with_default_encoders_and_model):
    app = make_flask_app(database_with_default_encoders_and_model)
    yield Client(CFG.server.uri, TestRequest(app.test_client()))


@pytest.fixture
def test_collection():
    collection_name = str(uuid.uuid4())
    return Collection(identifier=collection_name)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_add_load(client, test_db, test_collection):
    m = TorchModel(
        identifier='test-add-client',
        object=torch.nn.Linear(10, 20),
        encoder=tensor(torch.float, shape=(20,)),
    )
    client.add(m)
    models = test_db.show('model')
    print(models)
    assert 'test-add-client' in models
    m = client.load('model', 'test-add-client')
    assert isinstance(m.object.artifact, torch.nn.Module)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_show(client, database_with_default_encoders_and_model):
    encoders = client.show('encoder')
    assert encoders == ['torch.float32[16]', 'torch.float32[32]']


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_insert(
    client, database_with_default_encoders_and_model, test_collection, fake_inserts
):
    client.execute(test_collection.insert_many([fake_inserts[0]]))
    r = database_with_default_encoders_and_model.execute(test_collection.find_one())
    assert all(torch.eq(r['x'].x, fake_inserts[0]['x'].x))


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_remove(client, database_with_default_encoders_and_model):
    database_with_default_encoders_and_model.add(tensor(torch.float64, shape=(32,)))
    encoders = client.show('encoder')
    assert encoders == ['torch.float32[16]', 'torch.float32[32]', 'torch.float64[32]']

    client.remove('encoder', 'torch.float64[32]', force=True)
    encoders = client.show('encoder')
    assert encoders == ['torch.float32[16]', 'torch.float32[32]']


@pytest.mark.skipif(not torch, reason='Torch not installed')
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
