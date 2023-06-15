from contextlib import contextmanager
from typing import Iterator
from unittest import mock

import pytest

from superduperdb.datalayer.mongodb.client import SuperDuperClient
from superduperdb.vector_search.base import VectorDatabase

from .conftest_mongodb import MongoDBConfig as TestMongoDBConfig
from superduperdb.misc.config import (
    Config as SuperDuperConfig,
    VectorSearchConfig,
    MilvusConfig,
    MongoDB as MongoDBConfig,
)

pytest_plugins = [
    "tests.conftest_mongodb",
    "tests.integration.conftest_milvus",
]


@contextmanager
def create_client(*, mongodb_config: TestMongoDBConfig) -> Iterator[SuperDuperClient]:
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
def client(mongodb_server: TestMongoDBConfig) -> Iterator[SuperDuperClient]:
    with create_client(mongodb_config=mongodb_server) as client:
        yield client


@pytest.fixture(autouse=True)
def config(mongodb_server: TestMongoDBConfig) -> Iterator[None]:
    mongodb_config = MongoDBConfig(
        host=mongodb_server.host,
        port=mongodb_server.port,
        user=mongodb_server.username,
        password=mongodb_server.password,
    )
    with mock.patch('superduperdb.CFG.mongodb', mongodb_config):
        yield


@pytest.fixture
def config_mongodb_milvus(
    config: SuperDuperConfig, milvus_config: MilvusConfig
) -> Iterator[None]:
    vector_search_config = VectorSearchConfig(
        milvus=milvus_config,
    )
    with mock.patch('superduperdb.CFG.vector_search', vector_search_config):
        with VectorDatabase.create(
            config=vector_search_config
        ).init() as vector_database:
            with mock.patch(
                'superduperdb.datalayer.base.database.VECTOR_DATABASE',
                vector_database,
            ):
                yield
