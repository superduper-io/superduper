from contextlib import contextmanager
from dataclasses import asdict
from typing import Iterator
from unittest import mock
from pymongo import MongoClient

import pytest

from superduperdb.vector_search.base import VectorDatabase
from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.misc.config import DataLayer

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
    datalayer_cfg = DataLayer(
        data_backend_kwargs=asdict(TestMongoDBConfig()),
        data_backend_name='test_db',
        metadata_kwargs=asdict(TestMongoDBConfig()),
        metadata_name='test_db',
        artifact_store_kwargs=asdict(TestMongoDBConfig()),
        artifact_store_name='_filesystem:test_db',
    )

    with mock.patch('superduperdb.CFG.datalayer', datalayer_cfg):
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
