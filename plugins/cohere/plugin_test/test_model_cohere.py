import os

import pytest
import vcr

from superduper_cohere import CohereEmbed, CohereGenerate

CASSETTE_DIR = os.path.join(os.path.dirname(__file__), 'cassettes')


if os.getenv('COHERE_API_KEY') is None:
    mp = pytest.MonkeyPatch()
    mp.setenv('COHERE_API_KEY', 'sk-TopSecret')


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_embed_one.yaml',
    filter_headers=['authorization'],
)
def test_embed_one():
    embed = CohereEmbed(identifier='embed-english-v2.0')
    resp = embed.predict('Hello world')

    assert isinstance(resp, list)
    assert all(isinstance(x, float) for x in resp)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_embed_batch.yaml',
    filter_headers=['authorization'],
)
def test_embed_batch():
    embed = CohereEmbed(identifier='embed-english-v2.0', batch_size=1)
    resp = embed.predict_batches(['Hello', 'world'])

    assert len(resp) == 2
    assert isinstance(resp[0], list)
    assert all(isinstance(x, float) for x in resp[0])


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_generate.yaml',
    filter_headers=['authorization'],
)
def test_generate():
    e = CohereGenerate(identifier='base-light', prompt='Hello, {context}')
    resp = e.predict('', context=['world!'])

    assert isinstance(resp, str)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_batch_generate.yaml',
    filter_headers=['authorization'],
)
def test_batch_generate():
    e = CohereGenerate(identifier='base-light')
    resp = e.predict_batches(
        [
            (('Hello, world!',), {}),
        ]
    )

    assert isinstance(resp, list)
    assert isinstance(resp[0], str)
