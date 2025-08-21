import os
import pytest
import vcr
from superduper_groq import GroqChatCompletions
from dotenv import load_dotenv

load_dotenv()
CASSETTE_DIR = os.path.join(os.path.dirname(__file__), 'cassettes')

# Ensure API key is set for testing
if os.getenv('GROQ_API_KEY') is None:
    mp = pytest.MonkeyPatch()
    mp.setenv('GROQ_API_KEY', 'sk-TopSecret')

# -----------------------
# Single prediction test
# -----------------------
@vcr.use_cassette(
    f"{CASSETTE_DIR}/test_chat_predict.yaml",
    filter_headers=["authorization"],
    record_mode="all"
)
def test_chat_predict():
    model = GroqChatCompletions(
        identifier="llama3-8b-8192",
        system_message="You are a helpful assistant."
    )
    resp = model.predict("Hello world")
    assert isinstance(resp, str)

# -----------------------
# Prediction with context
# -----------------------
@vcr.use_cassette(
    f"{CASSETTE_DIR}/test_chat_predict_with_context.yaml",
    filter_headers=["authorization"],
    record_mode="all"
)
def test_chat_predict_with_context():
    model = GroqChatCompletions(
        identifier="llama3-8b-8192",
        system_message="You are a helpful assistant."
    )
    resp = model.predict("Tell me about", context=["AI"])
    assert isinstance(resp, str)

# -----------------------
# Batch prediction test
# -----------------------
@vcr.use_cassette(
    f"{CASSETTE_DIR}/test_chat_predict_batches.yaml",
    filter_headers=["authorization"],
    record_mode="all"
)
def test_chat_predict_batches():
    model = GroqChatCompletions(
        identifier="llama3-8b-8192",
        system_message="You are a helpful assistant."
    )
    dataset = ["Hello", "Summarize AI history in 2 sentences"]
    resp = model.predict_batches(dataset)
    assert isinstance(resp, list)
    assert all(isinstance(r, str) for r in resp)
