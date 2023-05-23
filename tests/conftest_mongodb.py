from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import pymongo
import pytest
from tenacity import RetryError, Retrying, stop_after_delay


@dataclass(frozen=True)
class MongoDBConfig:
    host: str = "localhost"
    port: int = 27017
    username: str = field(repr=False, default="testmongodbuser")
    password: str = field(repr=False, default="testmongodbpassword")
    server_selection_timeout_s: float = 5.0


@contextmanager
def create_mongodb_client(config: MongoDBConfig) -> Iterator[pymongo.MongoClient]:
    client: pymongo.MongoClient
    with pymongo.MongoClient(
        host=config.host,
        port=config.port,
        username=config.username,
        password=config.password,
        serverSelectionTimeoutMS=int(config.server_selection_timeout_s * 1000),
    ) as client:
        yield client


def wait_for_mongodb(config: MongoDBConfig, *, timeout_s: float = 30) -> None:
    try:
        for attempt in Retrying(stop=stop_after_delay(timeout_s)):
            with attempt:
                with create_mongodb_client(config) as client:
                    client.is_mongos
                    return
            print("Waiting for mongodb to start...")
    except RetryError:
        pytest.fail("Could not connect to mongodb")


def cleanup_mongodb(config: MongoDBConfig) -> None:
    with create_mongodb_client(config) as client:
        for database_name in client.list_database_names():
            if database_name in ("admin", "config", "local"):
                continue
            client.drop_database(database_name)


@pytest.fixture(scope='session')
def mongodb_config() -> MongoDBConfig:
    return MongoDBConfig()


@pytest.fixture(scope='session')
def _mongodb_server(mongodb_config: MongoDBConfig) -> Iterator[MongoDBConfig]:
    wait_for_mongodb(mongodb_config)
    yield mongodb_config


@pytest.fixture
def mongodb_server(_mongodb_server: MongoDBConfig) -> Iterator[MongoDBConfig]:
    # we are cleaning up the database before each test because in case of a test failure
    # one might want to inspect the state of the database
    cleanup_mongodb(_mongodb_server)
    yield _mongodb_server


@pytest.fixture
def mongodb_client(mongodb_server: MongoDBConfig) -> Iterator[pymongo.MongoClient]:
    with create_mongodb_client(mongodb_server) as client:
        yield client
