import os
from test.unittest.ext.llm.utils import check_llm_as_listener_model, check_predict
from typing import Iterator

import pytest
import vcr
from superduper import superduper
from superduper.base.datalayer import Datalayer

from superduper_vllm.model import VllmAPI

CASSETTE_DIR = os.path.join(os.path.dirname(__file__), 'cassettes')

api_url = "http://localhost:8000/generate"


@pytest.fixture
def db() -> Iterator[Datalayer]:
    db = superduper("mongomock://test_db")

    yield db
    db.drop(force=True, data=True)


@pytest.fixture
def openai_mock(monkeypatch):
    if os.getenv("OPENAI_API_KEY") is None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-TopSecret")


@vcr.use_cassette(
    f"{CASSETTE_DIR}/test_predict_api.yaml",
    filter_headers=["authorization"],
    record_on_exception=False,
    ignore_localhost=False,
)
def test_predict_api(db):
    """Test chat."""
    check_predict(db, VllmAPI(identifier="llm", api_url=api_url, max_batch_size=1))


@vcr.use_cassette(
    f"{CASSETTE_DIR}/test_llm_as_listener_model_api.yaml",
    filter_headers=["authorization"],
    record_on_exception=False,
    ignore_localhost=False,
)
def test_llm_as_listener_model_api(db):
    # vcr does not support multiple requests in a single test, set max_batch_size to 1
    check_llm_as_listener_model(
        db, VllmAPI(identifier="llm", api_url=api_url, max_batch_size=1)
    )
