import os

import pytest
import vcr

from superduper_anthropic import AnthropicCompletions

CASSETTE_DIR = os.path.join(os.path.dirname(__file__), 'cassettes')


@pytest.mark.skip(reason="API is not publicly available yet")
@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_completions.yaml',
    filter_headers=['authorization'],
)
def test_completions():
    e = AnthropicCompletions(identifier='claude-2.1', predict_kwargs={'max_tokens': 64})
    resp = e.predict('Hello')
    assert isinstance(resp, str)


@pytest.mark.skip(reason="API is not publicly available yet")
@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_batch_completions.yaml',
    filter_headers=['authorization'],
)
def test_batch_completions():
    e = AnthropicCompletions(identifier='claude-2.1', predict_kwargs={'max_tokens': 64})
    resp = e.predict_batches(['Hello, world!'])

    assert isinstance(resp, list)
    assert isinstance(resp[0], str)
