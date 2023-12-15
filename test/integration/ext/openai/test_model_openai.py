import io
import os

import openai
import pytest
import vcr
from vcr.stubs import httpx_stubs

from superduperdb.ext.openai.model import (
    OpenAIAudioTranscription,
    OpenAIAudioTranslation,
    OpenAIChatCompletion,
    OpenAIEmbedding,
    OpenAIImageCreation,
    OpenAIImageEdit,
    _available_models,
)

PNG_BYTE_SIGNATURE = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'

CASSETTE_DIR = 'test/integration/ext/openai/cassettes'

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
    '''
    VCR filter function to only record the PNG signature in the response.

    This is necessary because the response is a PNG which can be quite large.
    '''
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

    assert len(resp) == e.shape[0]
    assert all(isinstance(x, float) for x in resp)


@vcr.use_cassette()
def test_batch_embed():
    e = OpenAIEmbedding(identifier='text-embedding-ada-002')
    resp = e.predict(['Hello', 'world!'], batch_size=1)

    assert len(resp) == 2
    assert all(len(x) == e.shape[0] for x in resp)
    assert all(isinstance(x, float) for y in resp for x in y)


@vcr.use_cassette()
@pytest.mark.asyncio
async def test_embed_async():
    e = OpenAIEmbedding(identifier='text-embedding-ada-002')
    resp = await e.apredict('Hello, world!')

    assert len(resp) == e.shape[0]
    assert all(isinstance(x, float) for x in resp)


@vcr.use_cassette()
@pytest.mark.asyncio
async def test_batch_embed_async():
    e = OpenAIEmbedding(identifier='text-embedding-ada-002')
    resp = await e.apredict(['Hello', 'world!'], batch_size=1)

    assert len(resp) == 2
    assert all(len(x) == e.shape[0] for x in resp)
    assert all(isinstance(x, float) for y in resp for x in y)


@vcr.use_cassette()
def test_chat():
    e = OpenAIChatCompletion(identifier='gpt-3.5-turbo', prompt='Hello, {context}')
    resp = e.predict('', one=True, context=['world!'])

    assert isinstance(resp, str)


@vcr.use_cassette()
def test_batch_chat():
    e = OpenAIChatCompletion(identifier='gpt-3.5-turbo')
    resp = e.predict(['Hello, world!'], one=False)

    assert isinstance(resp, list)
    assert isinstance(resp[0], str)


@vcr.use_cassette()
@pytest.mark.asyncio
async def test_chat_async():
    e = OpenAIChatCompletion(identifier='gpt-3.5-turbo', prompt='Hello, {context}')
    resp = await e.apredict('', one=True, context=['world!'])

    assert isinstance(resp, str)


@vcr.use_cassette()
@pytest.mark.asyncio
async def test_batch_chat_async():
    e = OpenAIChatCompletion(identifier='gpt-3.5-turbo')
    resp = await e.apredict(['Hello, world!'], one=False)

    assert isinstance(resp, list)
    assert isinstance(resp[0], str)


@vcr.use_cassette()
def test_create_url():
    e = OpenAIImageCreation(
        identifier='dall-e',
        prompt='a close up, studio photographic portrait of a {context}',
    )
    resp = e.predict('', one=True, response_format='url', context=['cat'])

    # PNG 8-byte signature
    assert resp[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette()
def test_create_url_batch():
    e = OpenAIImageCreation(
        identifier='dall-e', prompt='a close up, studio photographic portrait of a'
    )
    resp = e.predict(['cat', 'dog'], response_format='url')

    for img in resp:
        # PNG 8-byte signature
        assert img[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette()
@pytest.mark.asyncio
async def test_create_async():
    e = OpenAIImageCreation(
        identifier='dall-e',
        prompt='a close up, studio photographic portrait of a {context}',
    )
    resp = await e.apredict('', one=True, context=['cat'])

    assert isinstance(resp, bytes)

    # PNG 8-byte signature
    assert resp[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette()
@pytest.mark.asyncio
async def test_create_url_async():
    e = OpenAIImageCreation(
        identifier='dall-e',
        prompt='a close up, studio photographic portrait of a {context}',
    )
    resp = await e.apredict('', one=True, response_format='url', context=['cat'])

    # PNG 8-byte signature
    assert resp[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette()
@pytest.mark.asyncio
async def test_create_url_async_batch():
    e = OpenAIImageCreation(
        identifier='dall-e', prompt='a close up, studio photographic portrait of a'
    )
    resp = await e.apredict(['cat', 'dog'], response_format='url')

    for img in resp:
        # PNG 8-byte signature
        assert img[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette()
def test_edit_url():
    e = OpenAIImageEdit(
        identifier='dall-e', prompt='A celebration party at the launch of {context}'
    )
    with open('test/material/data/rickroll.png', 'rb') as f:
        buffer = io.BytesIO(f.read())
    resp = e.predict(buffer, one=True, response_format='url', context=['superduperdb'])
    buffer.close()

    # PNG 8-byte signature
    assert resp[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette()
def test_edit_url_batch():
    e = OpenAIImageEdit(
        identifier='dall-e', prompt='A celebration party at the launch of superduperdb'
    )
    with open('test/material/data/rickroll.png', 'rb') as f:
        buffer_one = io.BytesIO(f.read())
    with open('test/material/data/rickroll.png', 'rb') as f:
        buffer_two = io.BytesIO(f.read())

    resp = e.predict([buffer_one, buffer_two], response_format='url')

    buffer_one.close()
    buffer_two.close()

    for img in resp:
        # PNG 8-byte signature
        assert img[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette()
@pytest.mark.asyncio
async def test_edit_async():
    e = OpenAIImageEdit(
        identifier='dall-e', prompt='A celebration party at the launch of {context}'
    )
    with open('test/material/data/rickroll.png', 'rb') as f:
        buffer = io.BytesIO(f.read())
    resp = await e.apredict(buffer, one=True, context=['superduperdb'])
    buffer.close()

    assert isinstance(resp, bytes)

    # PNG 8-byte signature
    assert resp[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette()
@pytest.mark.asyncio
async def test_edit_url_async():
    e = OpenAIImageEdit(
        identifier='dall-e', prompt='A celebration party at the launch of {context}'
    )
    with open('test/material/data/rickroll.png', 'rb') as f:
        buffer = io.BytesIO(f.read())
    resp = await e.apredict(
        buffer, one=True, response_format='url', context=['superduperdb']
    )
    buffer.close()

    # PNG 8-byte signature
    assert resp[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette()
@pytest.mark.asyncio
async def test_edit_url_async_batch():
    e = OpenAIImageEdit(
        identifier='dall-e', prompt='A celebration party at the launch of superduperdb'
    )
    with open('test/material/data/rickroll.png', 'rb') as f:
        buffer_one = io.BytesIO(f.read())
    with open('test/material/data/rickroll.png', 'rb') as f:
        buffer_two = io.BytesIO(f.read())

    resp = await e.apredict([buffer_one, buffer_two], response_format='url')

    buffer_one.close()
    buffer_two.close()

    for img in resp:
        # PNG 8-byte signature
        assert img[0:16] == PNG_BYTE_SIGNATURE


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
    resp = e.predict(buffer, one=True, context=['United States'])
    buffer.close()

    assert 'United States' in resp


@vcr.use_cassette()
def test_batch_transcribe():
    with open('test/material/data/test.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
        buffer.name = 'test.wav'

    with open('test/material/data/test.wav', 'rb') as f:
        buffer2 = io.BytesIO(f.read())
        buffer2.name = 'test.wav'

    e = OpenAIAudioTranscription(identifier='whisper-1')
    resp = e.predict([buffer, buffer2], one=False, batch_size=1)
    buffer.close()

    assert len(resp) == 2
    assert resp[0] == resp[1]
    assert 'United States' in resp[0]


@vcr.use_cassette()
@pytest.mark.asyncio
async def test_transcribe_async():
    with open('test/material/data/test.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
    buffer.name = 'test.wav'
    prompt = (
        'i have some advice for you. write all text in lower-case.'
        'only make an exception for the following words: {context}'
    )
    e = OpenAIAudioTranscription(identifier='whisper-1', prompt=prompt)
    resp = await e.apredict(buffer, one=True, context=['United States'])
    buffer.close()

    assert 'United States' in resp


@vcr.use_cassette()
@pytest.mark.asyncio
async def test_batch_transcribe_async():
    with open('test/material/data/test.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
        buffer.name = 'test1.wav'

    with open('test/material/data/test.wav', 'rb') as f:
        buffer2 = io.BytesIO(f.read())
        buffer2.name = 'test1.wav'
    e = OpenAIAudioTranscription(identifier='whisper-1')

    resp = await e.apredict([buffer, buffer2], one=False, batch_size=1)
    buffer.close()

    assert len(resp) == 2
    assert resp[0] == resp[1]
    assert 'United States' in resp[0]


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
    resp = e.predict(buffer, one=True, context=['Emmerich'])
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

    e = OpenAIAudioTranslation(identifier='whisper-1')
    resp = e.predict([buffer, buffer2], one=False, batch_size=1)
    buffer.close()

    assert len(resp) == 2
    assert resp[0] == resp[1]
    assert 'station' in resp[0]


@vcr.use_cassette()
@pytest.mark.asyncio
async def test_translate_async():
    with open('test/material/data/german.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
    buffer.name = 'test.wav'
    prompt = (
        'i have some advice for you. write all text in lower-case.'
        'only make an exception for the following words: {context}'
    )
    e = OpenAIAudioTranslation(identifier='whisper-1', prompt=prompt)
    resp = await e.apredict(buffer, one=True, context=['Emmerich'])
    buffer.close()

    assert 'station' in resp


@vcr.use_cassette()
@pytest.mark.asyncio
async def test_batch_translate_async():
    with open('test/material/data/german.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
        buffer.name = 'test1.wav'

    with open('test/material/data/german.wav', 'rb') as f:
        buffer2 = io.BytesIO(f.read())
        buffer2.name = 'test1.wav'
    e = OpenAIAudioTranslation(identifier='whisper-1')

    resp = await e.apredict([buffer, buffer2], one=False, batch_size=1)
    buffer.close()

    assert len(resp) == 2
    assert resp[0] == resp[1]
    assert 'station' in resp[0]
