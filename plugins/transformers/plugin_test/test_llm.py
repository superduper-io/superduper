from test.unittest.ext.llm.utils import check_llm_as_listener_model, check_predict

from superduper_transformers import LLM

TEST_MODEL_NAME = "facebook/opt-125m"


def test_predict(db):
    """Test chat."""
    model = LLM(identifier="llm", model_name_or_path=TEST_MODEL_NAME)

    check_predict(db, model)


def test_model_as_listener_model(db):
    model = LLM(identifier="llm", model_name_or_path=TEST_MODEL_NAME, datatype='str')
    check_llm_as_listener_model(db, model)
