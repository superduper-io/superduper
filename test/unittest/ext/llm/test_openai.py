import os
from test.db_config import DBConfig
from test.unittest.ext.llm.utils import check_llm_as_listener_model, check_predict

import pytest
import vcr

from superduperdb.ext.openai.model import OpenAILLM

CASSETTE_DIR = "test/unittest/ext/cassettes/llm/openai"


@pytest.fixture
def openai_mock(monkeypatch):
    if os.getenv("OPENAI_API_KEY") is None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-TopSecret")


@pytest.mark.skip("Skip until openai connection failures fixed")
@vcr.use_cassette(
    f"{CASSETTE_DIR}/test_predict.yaml",
    filter_headers=["authorization"],
    record_on_exception=False,
    ignore_localhost=True,
)
@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_predict(db, openai_mock):
    """Test chat."""
    check_predict(db, OpenAILLM(model_name="gpt-3.5-turbo"))
    check_predict(
        db, OpenAILLM(identifier="chat-llm", model_name="gpt-3.5-turbo", chat=True)
    )


@pytest.mark.skip("Skip until openai connection failures fixed")
@vcr.use_cassette(
    f"{CASSETTE_DIR}/test_llm_as_listener_model.yaml",
    filter_headers=["authorization"],
    record_on_exception=False,
    ignore_localhost=True,
)
@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_llm_as_listener_model(db, openai_mock):
    check_llm_as_listener_model(db, OpenAILLM(model_name="gpt-3.5-turbo"))
