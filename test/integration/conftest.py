import os
import random
import time
from threading import Thread
from unittest import mock

import numpy as np
import pytest
import torch

from superduperdb import CFG
from superduperdb.base.config import DataLayer, DataLayers
from superduperdb.container.document import Document
from superduperdb.db.base.build import build_datalayer
from superduperdb.ext.torch.model import TorchModel
from superduperdb.ext.torch.tensor import tensor
from superduperdb.server.dask_client import dask_client
from superduperdb.server.server import serve

'''
Fixtures included here can create:
- a MongoDB client
- a MongoDB collection with some basic data
- a local Dask client
- a local SuperDuperDB server linked to the MongoDB client
- INSERT OTHERS HERE

When adding new fixtures, please be considerate when building on top of other
fixtures to create deeply nested fixtures. This will make it easier to understand
the test suite for others.
'''

# Set the seeds
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


@pytest.fixture()
def database_with_default_encoders_and_model(empty_database):
    empty_database.add(tensor(torch.float, shape=(32,)))
    empty_database.add(tensor(torch.float, shape=(16,)))
    empty_database.add(
        TorchModel(
            object=torch.nn.Linear(32, 16),
            identifier='model_linear_a',
            encoder='torch.float32[16]',
        )
    )
    yield empty_database

    empty_database.remove('model', 'model_linear_a', force=True)
    empty_database.remove('encoder', 'torch.float32[16]', force=True)
    empty_database.remove('encoder', 'torch.float32[32]', force=True)


def fake_tensor_data(encoder, update: bool = True):
    data = []
    for i in range(10):
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


@pytest.fixture()
def fake_inserts(database_with_default_encoders_and_model):
    encoder = database_with_default_encoders_and_model.encoders['torch.float32[32]']
    return fake_tensor_data(encoder, update=False)


@pytest.fixture()
def fake_updates(database_with_default_encoders_and_model):
    encoder = database_with_default_encoders_and_model.encoders['torch.float32[32]']
    return fake_tensor_data(encoder, update=True)


# We only want to start the server once ie session scope. This is a
# database fixture with session scope that can be used with the server.
@pytest.fixture(scope="session")
def session_database(mongodb_test_config, mongodb_client):
    database_name = "test_session_scope_database"
    data_layers_cfg = DataLayers(
        artifact=DataLayer(
            name=f'_filesystem:{database_name}', kwargs=mongodb_test_config
        ),
        data_backend=DataLayer(name=database_name, kwargs=mongodb_test_config),
        metadata=DataLayer(name=database_name, kwargs=mongodb_test_config),
    )

    with mock.patch('superduperdb.CFG.data_layers', data_layers_cfg):
        session_database = build_datalayer(pymongo=mongodb_client)
        session_database.add(tensor(torch.float, shape=(32,)))
        session_database.add(tensor(torch.float, shape=(16,)))
        session_database.add(
            TorchModel(
                object=torch.nn.Linear(32, 16),
                identifier='model_linear_a',
                encoder='torch.float32[16]',
            )
        )
        yield session_database

    # clean-up the databases created by build_datalayer
    mongodb_client.drop_database(f'_filesystem:{database_name}')
    mongodb_client.drop_database(f'{database_name}')


@pytest.fixture(scope="session")
def test_server(session_database):
    app = serve(session_database)
    t = Thread(
        target=app.run,
        kwargs={"host": CFG.server.host, "port": CFG.server.port},
        daemon=True,
    )
    t.start()
    time.sleep(2)
    yield


@pytest.fixture()
def local_dask_client(database_name):
    for component in ['DATA_BACKEND', 'ARTIFACT', 'METADATA']:
        os.environ[f'SUPERDUPERDB_DATA_LAYERS_{component}_KWARGS_PORT'] = '27018'
        os.environ[f'SUPERDUPERDB_DATA_LAYERS_{component}_KWARGS_HOST'] = 'localhost'
        os.environ[
            f'SUPERDUPERDB_DATA_LAYERS_{component}_KWARGS_USERNAME'
        ] = 'testmongodbuser'
        os.environ[
            f'SUPERDUPERDB_DATA_LAYERS_{component}_KWARGS_PASSWORD'
        ] = 'testmongodbpassword'

        os.environ[f'SUPERDUPERDB_DATA_LAYERS_{component}_NAME'] = (
            f'_filesystem:{database_name}'
            if component == "ARTIFACT"
            else f'{database_name}'
        )
    client = dask_client(CFG.dask, local=True)
    yield client
    client.shutdown()
