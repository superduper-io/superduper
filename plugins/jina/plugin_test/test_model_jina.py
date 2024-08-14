import os

import pytest
import vcr

from superduper_jina import JinaEmbedding

CASSETTE_DIR = os.path.join(os.path.dirname(__file__), 'cassettes')


if os.getenv('JINA_API_KEY') is None:
    mp = pytest.MonkeyPatch()
    mp.setenv('JINA_API_KEY', 'sk-TopSecret')


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_embed_one.yaml',
    filter_headers=['Authorization'],
)
def test_embed_one():
    embed = JinaEmbedding(identifier='jina-embeddings-v2-base-en')
    resp = embed.predict('Hello world')

    assert len(resp) == embed.shape[0]
    assert isinstance(resp, list)
    assert all(isinstance(x, float) for x in resp)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_embed_batch.yaml',
    filter_headers=['Authorization'],
)
def test_embed_batch():
    embed = JinaEmbedding(identifier='jina-embeddings-v2-base-en', batch_size=3)
    resp = embed.predict_batches(['Hello', 'world', 'I', 'am', 'here'])

    assert len(resp) == 5
    assert len(resp[0]) == embed.shape[0]
    assert isinstance(resp[0], list)
    assert all(isinstance(x, float) for x in resp[0])
