from contextlib import contextmanager
from typing import Iterator
from unittest import mock

import pytest

from superduperdb.datalayer.mongodb.client import SuperDuperClient

from .conftest_mongodb import MongoDBConfig
from superduperdb.misc import config as _config

pytest_plugins = [
    "tests.conftest_mongodb",
]


@contextmanager
def create_client(*, mongodb_config: MongoDBConfig) -> Iterator[SuperDuperClient]:
    client = SuperDuperClient(
        host=mongodb_config.host,
        port=mongodb_config.port,
        username=mongodb_config.username,
        password=mongodb_config.password,
        serverSelectionTimeoutMS=int(mongodb_config.server_selection_timeout_s * 1000),
    )
    # avoiding a typing error here as SuperDuperClient inherits from pymongo.MongoClient
    # which returns a generic MongoClient[...] object in __enter__, but we still want to
    # keep the SuperDuperClient type
    with client:
        yield client


@pytest.fixture
def client(mongodb_server: MongoDBConfig) -> Iterator[SuperDuperClient]:
    with create_client(mongodb_config=mongodb_server) as client:
        yield client


@pytest.fixture(autouse=True)
def config(mongodb_server: MongoDBConfig) -> Iterator[None]:
    our_config = _config.MongoDB(
        host=mongodb_server.host,
        port=mongodb_server.port,
        user=mongodb_server.username,
        password=mongodb_server.password,
    )

    with mock.patch('superduperdb.CFG.mongodb', our_config):
        yield
