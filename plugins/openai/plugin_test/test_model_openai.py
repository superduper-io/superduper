import io
import os

import openai
import pytest
import vcr
from vcr.stubs import httpx_stubs

from superduper_openai.model import (
    OpenAIAudioTranscription,
    OpenAIAudioTranslation,
    OpenAIChatCompletion,
    OpenAIEmbedding,
    _available_models,
)

PNG_BYTE_SIGNATURE = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'

CASSETTE_DIR = os.path.join(os.path.dirname(__file__), 'cassettes')

if os.getenv('OPENAI_API_KEY') is None:
    mp = pytest.MonkeyPatch()
    mp.setattr(openai, 'api_key', 'sk-TopSecret')
    mp.setenv('OPENAI_API_KEY', 'sk-TopSecret')


# monkey patch vcr to make it work with upload binary data
def _make_vcr_request(httpx_request, **kwargs):
    from vcr.request import Request as VcrRequest

    try:
        body = httpx_request.read().decode("utf-8")
    except UnicodeDecodeError:
        body = str(httpx_request.read())
    uri = str(httpx_request.url)
    headers = dict(httpx_request.headers)
    return VcrRequest(httpx_request.method, uri, body, headers)


httpx_stubs._make_vcr_request = _make_vcr_request


def before_record_response(response):
    """
    VCR filter function to only record the PNG signature in the response.

    This is necessary because the response is a PNG which can be quite large.
    """
    if 'body' not in response:
        return response
    if PNG_BYTE_SIGNATURE in response['body']['string']:
        response['body']['string'] = PNG_BYTE_SIGNATURE

    if b'b64_json' in response['body']['string']:
        # "b64_json": base64.b64encode(PNG_BYTE_SIGNATURE)
        response['body']['string'] = (
            b'{\n  \"created\": 1695982982,\n  \"data\": '
            b'[\n    {\n      \"b64_json\": \"iVBORw0KGgoAAAANSUhEUg==\" } ]}'
        )
    return response


def before_record_request(request):
    # make the cassette simpler
    if getattr(request, 'body', None) is not None:
        request.body = 'fake_body'
    # make the cassette simpler
    request.headers = {}
    return request


# TODO: Move to top level of test dir, help other api tests
vcr = vcr.VCR(
    path_transformer=lambda x: x + '.yaml',
    match_on=('method', 'path'),
    filter_headers=['authorization'],
    cassette_library_dir=CASSETTE_DIR,
    before_record_request=before_record_request,
    before_record_response=before_record_response,
    record_on_exception=False,
)


@pytest.fixture(autouse=True)
def mock_lru_cache():
    _available_models.cache_clear()


@vcr.use_cassette()
def test_embed():
    e = OpenAIEmbedding(identifier='text-embedding-ada-002')
    resp = e.predict('Hello, world!')

    assert str(resp.dtype) == 'float32'


@vcr.use_cassette()
def test_batch_embed():
    e = OpenAIEmbedding(identifier='text-embedding-ada-002', batch_size=1)
    resp = e.predict_batches(['Hello', 'world!'])

    assert len(resp) == 2
    assert all(str(x.dtype) == 'float32' for x in resp)


@vcr.use_cassette()
def test_chat():
    e = OpenAIChatCompletion(identifier='gpt-3.5-turbo', prompt='Hello, {context}')
    resp = e.predict('', context=['world!'])

    assert isinstance(resp, str)


@vcr.use_cassette()
def test_batch_chat():
    e = OpenAIChatCompletion(identifier='gpt-3.5-turbo')
    resp = e.predict_batches([(('Hello, world!',), {})])

    assert isinstance(resp, list)
    assert isinstance(resp[0], str)


@vcr.use_cassette()
def test_transcribe():
    with open('test/material/data/test.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
    buffer.name = 'test.wav'
    prompt = (
        'i have some advice for you. write all text in lower-case.'
        'only make an exception for the following words: {context}'
    )
    e = OpenAIAudioTranscription(identifier='whisper-1', prompt=prompt)
    resp = e.predict(buffer, context=['United States'])
    buffer.close()

    assert 'United States' in resp


@vcr.use_cassette()
def test_translate():
    with open('test/material/data/german.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
    buffer.name = 'test.wav'
    prompt = (
        'i have some advice for you. write all text in lower-case.'
        'only make an exception for the following words: {context}'
    )
    e = OpenAIAudioTranslation(identifier='whisper-1', prompt=prompt)
    resp = e.predict(buffer, context=['Emmerich'])
    buffer.close()

    assert 'station' in resp


@vcr.use_cassette()
def test_batch_translate():
    with open('test/material/data/german.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
        buffer.name = 'test.wav'

    with open('test/material/data/german.wav', 'rb') as f:
        buffer2 = io.BytesIO(f.read())
        buffer2.name = 'test.wav'

    e = OpenAIAudioTranslation(identifier='whisper-1', batch_size=1)
    resp = e.predict_batches([buffer, buffer2])
    buffer.close()

    assert len(resp) == 2
    assert resp[0] == resp[1]
    assert 'station' in resp[0]
