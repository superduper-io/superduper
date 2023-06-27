from contextlib import contextmanager
from dataclasses import asdict
from typing import Iterator
from unittest import mock
from pymongo import MongoClient

import pytest
import superduperdb as s

from superduperdb.vector_search.base import VectorDatabase
from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.misc.config import DataLayer, DataLayers

from .conftest_mongodb import MongoDBConfig as TestMongoDBConfig
from superduperdb.misc.config import (
    Config as SuperDuperConfig,
    MongoDB as MongoDBConfig,
)

pytest_plugins = [
    "tests.conftest_mongodb",
    "tests.integration.conftest_milvus",
]


@contextmanager
def create_datalayer(*, mongodb_config: MongoDBConfig) -> Iterator[BaseDatabase]:
    from superduperdb.datalayer.base.build import build_datalayer

    mongo_client = MongoClient(
        host=mongodb_config.host,
        port=mongodb_config.port,
        username=mongodb_config.username,
        password=mongodb_config.password,
        serverSelectionTimeoutMS=int(mongodb_config.serverSelectionTimeoutMS * 1000),
    )
    with mongo_client:
        yield build_datalayer(
            pymongo=mongo_client,
        )


@pytest.fixture
def test_db(mongodb_server: MongoDBConfig) -> Iterator[BaseDatabase]:
    with create_datalayer(mongodb_config=mongodb_server) as db:
        yield db


@pytest.fixture(autouse=True)
def config(mongodb_server: MongoDBConfig) -> Iterator[None]:
    kwargs = asdict(TestMongoDBConfig())
    data_layers_cfg = DataLayers(
        artifact=DataLayer(name='_filesystem:test_db', kwargs=kwargs),
        data_backend=DataLayer(name='test_db', kwargs=kwargs),
        metadata=DataLayer(name='test_db', kwargs=kwargs),
    )

    with mock.patch('superduperdb.CFG.data_layers', data_layers_cfg):
        yield


@pytest.fixture
def config_mongodb_milvus(
    config: SuperDuperConfig, milvus_config: s.config.Milvus
) -> Iterator[None]:
    vector_search_config = s.config.VectorSearch(milvus=milvus_config)
    with mock.patch('superduperdb.CFG.vector_search', vector_search_config):
        with VectorDatabase.create(
            config=vector_search_config
        ).init() as vector_database:
            with mock.patch(
                'superduperdb.datalayer.base.database.VECTOR_DATABASE',
                vector_database,
            ):
                yield
