import os
import random

import numpy as np
import pytest

try:
    import torch

    from superduperdb.ext.torch.encoder import tensor
    from superduperdb.ext.torch.model import TorchModel
except ImportError:
    torch = None

from superduperdb.backends.dask.compute import DaskComputeBackend
from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.document import Document
from superduperdb.components.listener import Listener
from superduperdb.components.vector_index import VectorIndex

'''
All pytest fixtures with _package scope_ are defined in this module.
Package scope means that the fixture will be executed once per package,
which in this case means once per `test/integration/` directory.

Fixtures included here can create:
- a MongoDB client
- a MongoDB collection with some basic data
- a local Dask client
- a local SuperDuperDB server linked to the MongoDB client

When adding new fixtures, please try to avoid building on top of other fixtures
as much as possible. This will make it easier to understand the test suite.
'''

# Set the seeds
random.seed(42)
torch and torch.manual_seed(42)
np.random.seed(42)


def add_models_encoders(test_db):
    test_db.add(tensor(torch.float, shape=(32,)))
    test_db.add(tensor(torch.float, shape=(16,)))
    test_db.add(
        TorchModel(
            object=torch.nn.Linear(32, 16),
            identifier='model_linear_a',
            encoder='torch.float32[16]',
        )
    )
    test_db.add(
        Listener(
            select=Collection(identifier='documents').find(),
            key='x',
            model='model_linear_a',
        )
    )
    test_db.add(
        Listener(
            select=Collection(identifier='documents').find(),
            key='z',
            model='model_linear_a',
        )
    )
    vi = VectorIndex(
        identifier='test_index',
        indexing_listener='model_linear_a/x',
        compatible_listener='model_linear_a/z',
    )
    test_db.add(vi)
    return test_db


@pytest.fixture
def database_with_default_encoders_and_model(test_db):
    yield add_models_encoders(test_db)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def fake_tensor_data(encoder, update: bool = True):
    data = []
    for _ in range(10):
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


@pytest.fixture
def fake_inserts(database_with_default_encoders_and_model):
    encoder = database_with_default_encoders_and_model.encoders['torch.float32[32]']
    return fake_tensor_data(encoder, update=False)


@pytest.fixture
def fake_updates(database_with_default_encoders_and_model):
    encoder = database_with_default_encoders_and_model.encoders['torch.float32[32]']
    return fake_tensor_data(encoder, update=True)


@pytest.fixture
def dask_client(monkeypatch, request):
    db_name = "test_db"
    data_backend = f'mongodb://superduper:superduper@localhost:27017/{db_name}'

    data_backend = os.environ.get('SUPERDUPER_MONGO_URI', data_backend)
    address = os.environ.get('SUPERDUPER_DASK_URI', 'tcp://localhost:8786')

    monkeypatch.setenv('SUPERDUPERDB_DATA_BACKEND', data_backend)

    # Change the default value
    client = DaskComputeBackend(
        address=address,
        local=False,
    )

    yield client

    client.disconnect()


@pytest.fixture(scope='session')
def ray_client():
    # Change the default value
    from superduperdb.backends.ray.compute import RayComputeBackend

    address = os.environ.get('SUPERDUPER_RAY_URI', 'ray://127.0.0.1:10001')
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as working_dir:
        shutil.copytree('./test', os.path.join(working_dir, 'test'))
        client = RayComputeBackend(
            address=address,
            runtime_env={"working_dir": working_dir, "excludes": ["unittest"]},
        )
        yield client
        client.disconnect()
