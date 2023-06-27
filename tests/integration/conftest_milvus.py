import pytest
import superduperdb as s
import typing as t

from tenacity import RetryError, Retrying, stop_after_delay

from superduperdb.vector_search.milvus import MilvusClient


@pytest.fixture(scope="session")
def milvus_config() -> s.config.Milvus:
    return s.config.Milvus(
        host="localhost",
        port=19530,
        username="root",
        password="",
        consistency_level="Strong",
    )


def wait_for_milvus(config: s.config.Milvus, *, timeout_s: float = 30) -> None:
    try:
        for attempt in Retrying(stop=stop_after_delay(timeout_s)):
            with attempt:
                with MilvusClient(config=config).init():
                    return
            print("Waiting for milvus to start...")
    except RetryError:
        pytest.fail("Could not connect to milvus")


def cleanup_milvus(config: s.config.Milvus) -> None:
    with MilvusClient(config=config).init() as client:
        client.drop_all_collections()


@pytest.fixture(scope="session")
def _milvus_server(milvus_config: s.config.Milvus) -> t.Iterator[s.config.Milvus]:
    wait_for_milvus(milvus_config)
    yield milvus_config


@pytest.fixture
def milvus_server(_milvus_server: s.config.Milvus) -> t.Iterator[s.config.Milvus]:
    cleanup_milvus(_milvus_server)
    yield _milvus_server


@pytest.fixture
def milvus_client(milvus_server: s.config.Milvus) -> t.Iterator[MilvusClient]:
    with MilvusClient(config=milvus_server).init() as client:
        yield client
