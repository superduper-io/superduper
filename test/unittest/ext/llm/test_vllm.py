from test.db_config import DBConfig
from test.unittest.ext.llm.utils import check_llm_as_listener_model, check_predict

import pytest
import vcr

from superduperdb.ext.llm.vllm import VllmAPI, VllmOpenAI

CASSETTE_DIR = "test/unittest/ext/cassettes/llm/vllm"

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
openai_api_base = "http://ec2-54-208-130-192.compute-1.amazonaws.com:8000/v1"
api_url = "http://ec2-54-208-130-192.compute-1.amazonaws.com:8000/generate"


@vcr.use_cassette(
    f"{CASSETTE_DIR}/test_predict_openai.yaml",
    filter_headers=["authorization"],
    record_on_exception=False,
    ignore_localhost=True,
)
@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_predict_openai(db):
    """Test chat."""
    check_predict(
        db, VllmOpenAI(model_name=model_name, openai_api_base=openai_api_base)
    )
    check_predict(
        db,
        VllmOpenAI(
            identifier="chat-llm",
            model_name=model_name,
            openai_api_base=openai_api_base,
            chat=True,
        ),
    )


@vcr.use_cassette(
    f"{CASSETTE_DIR}/test_llm_as_listener_model_openai.yaml",
    filter_headers=["authorization"],
    record_on_exception=False,
    ignore_localhost=True,
)
@pytest.mark.parametrize("db", [DBConfig.sqldb_empty], indirect=True)
def test_llm_as_listener_model_openai(db):
    check_llm_as_listener_model(
        db,
        VllmOpenAI(model_name=model_name, openai_api_base=openai_api_base),
    )


@vcr.use_cassette(
    f"{CASSETTE_DIR}/test_predict_api.yaml",
    filter_headers=["authorization"],
    record_on_exception=False,
    ignore_localhost=True,
)
@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_predict_api(db):
    """Test chat."""
    check_predict(db, VllmAPI(identifier="llm", api_url=api_url))


@vcr.use_cassette(
    f"{CASSETTE_DIR}/test_llm_as_listener_model_openapi.yaml",
    filter_headers=["authorization"],
    record_on_exception=False,
    ignore_localhost=True,
)
@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_llm_as_listener_model_openapi(db):
    check_llm_as_listener_model(db, VllmAPI(identifier="llm", api_url=api_url))
