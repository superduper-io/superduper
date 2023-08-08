import uuid
from unittest import mock

import pytest
from pymongo import MongoClient
from tenacity import RetryError, Retrying, stop_after_delay

from superduperdb.base.config import DataLayer, DataLayers
from superduperdb.db.base.build import build_datalayer


@pytest.fixture(scope='session')
def mongodb_test_config():
    return {
        'host': '0.0.0.0',
        'port': 27018,
        'username': 'testmongodbuser',
        'password': 'testmongodbpassword',
        'serverSelectionTimeoutMS': 5000,
    }


@pytest.fixture(scope='session')
def mongodb_client(mongodb_test_config):
    mongo_client = MongoClient(**mongodb_test_config)

    try:
        for attempt in Retrying(stop=stop_after_delay(15)):
            with attempt:
                mongo_client.is_mongos
            break
    except RetryError:
        pytest.fail("Could not connect to mongodb")

    yield mongo_client

    mongo_client.close()


# Important: We assume that every test writes to a separate database
@pytest.fixture(scope="function")
def database_name():
    # ObjectId has no dashes - https://www.mongodb.com/docs/manual/reference/method/ObjectId/
    return str(uuid.uuid4()).replace("-", "")


@pytest.fixture(scope="function")
def empty_database(mongodb_test_config, mongodb_client, database_name):
    data_layers_cfg = DataLayers(
        artifact=DataLayer(
            name=f'_filesystem:{database_name}', kwargs=mongodb_test_config
        ),
        data_backend=DataLayer(name=database_name, kwargs=mongodb_test_config),
        metadata=DataLayer(name=database_name, kwargs=mongodb_test_config),
    )

    with mock.patch('superduperdb.CFG.data_layers', data_layers_cfg):
        yield build_datalayer(pymongo=mongodb_client)

    # clean-up the databases created by build_datalayer
    mongodb_client.drop_database(f'_filesystem:{database_name}')
    mongodb_client.drop_database(f'{database_name}')
