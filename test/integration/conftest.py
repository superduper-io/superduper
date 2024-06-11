import random

import numpy as np
import pytest

try:
    import torch

    from superduperdb.ext.torch.encoder import tensor
    from superduperdb.ext.torch.model import TorchModel
except ImportError:
    torch = None

from superduperdb.backends.mongodb.query import MongoQuery
from superduperdb.base.document import Document
from superduperdb.components.listener import Listener
from superduperdb.components.vector_index import VectorIndex

# Set the seeds
random.seed(42)
torch and torch.manual_seed(42)
np.random.seed(42)


def add_models_encoders(test_db):
    test_db.add(tensor(torch.float, shape=(32,)))
    _, dt_16 = test_db.add(tensor(torch.float, shape=(16,)))
    _, model = test_db.add(
        TorchModel(
            object=torch.nn.Linear(32, 16),
            identifier='model_linear_a',
            datatype=dt_16,
        )
    )
    _, indexing_listener = test_db.add(
        Listener(
            select=MongoQuery(table='documents').find(),
            key='x',
            model=model,
        )
    )
    _, compatible_listener = test_db.add(
        Listener(
            select=MongoQuery(table='documents').find(),
            key='z',
            model=model,
        )
    )
    vi = VectorIndex(
        identifier='test_index',
        indexing_listener=indexing_listener,
        compatible_listener=compatible_listener,
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
    dt = database_with_default_encoders_and_model.datatypes['torch.float32[32]']
    return fake_tensor_data(dt, update=False)


@pytest.fixture
def fake_updates(database_with_default_encoders_and_model):
    dt = database_with_default_encoders_and_model.datatypes['torch.float32[32]']
    return fake_tensor_data(dt, update=True)
