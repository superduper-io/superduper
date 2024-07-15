import pytest

from superduper.ext.transformers import LLM
from test.db_config import DBConfig
from test.unittest.ext.llm.utils import check_llm_as_listener_model, check_predict

TEST_MODEL_NAME = "facebook/opt-125m"


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_predict(db):
    """Test chat."""
    model = LLM(identifier="llm", model_name_or_path=TEST_MODEL_NAME)

    check_predict(db, model)


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_model_as_listener_model(db):
    model = LLM(identifier="llm", model_name_or_path=TEST_MODEL_NAME)
    check_llm_as_listener_model(db, model)
