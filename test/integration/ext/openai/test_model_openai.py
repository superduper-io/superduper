import io
import os

import openai
import pytest
import vcr
from superduperdb.ext.openai.model import (
    OpenAIAudioTranscription,
    OpenAIAudioTranslation,
    OpenAIChatCompletion,
    OpenAIEmbedding,
    OpenAIImageCreation,
    OpenAIImageEdit,
)

PNG_BYTE_SIGNATURE = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'

CASSETTE_DIR = 'test/integration/ext/openai/cassettes'

if os.getenv('OPENAI_API_KEY') is None:
    mp = pytest.MonkeyPatch()
    mp.setattr(openai, 'api_key', 'sk-TopSecret')
    mp.setenv('OPENAI_API_KEY', 'sk-TopSecret')


def _record_only_png_signature_in_response(response):
    '''
    VCR filter function to only record the PNG signature in the response.

    This is necessary because the response is a PNG which can be quite large.
    '''
    if PNG_BYTE_SIGNATURE in response['body']['string']:
        response['body']['string'] = PNG_BYTE_SIGNATURE

    if b'b64_json' in response['body']['string']:
        # "b64_json": base64.b64encode(PNG_BYTE_SIGNATURE)
        response['body']['string'] = (
            b'{\n  \"created\": 1695982982,\n  \"data\": '
            b'[\n    {\n      \"b64_json\": \"iVBORw0KGgoAAAANSUhEUg==\" } ]}'
        )
    return response


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_embed.yaml',
    filter_headers=['authorization'],
)
def test_embed():
    e = OpenAIEmbedding(model='text-embedding-ada-002')
    resp = e.predict('Hello, world!')

    assert len(resp) == e.shape[0]
    assert all(isinstance(x, float) for x in resp)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_batch_embed.yaml',
    filter_headers=['authorization'],
)
def test_batch_embed():
    e = OpenAIEmbedding(model='text-embedding-ada-002')
    resp = e.predict(['Hello', 'world!'], batch_size=1)

    assert len(resp) == 2
    assert all(len(x) == e.shape[0] for x in resp)
    assert all(isinstance(x, float) for y in resp for x in y)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_embed_async.yaml',
    filter_headers=['authorization'],
)
@pytest.mark.asyncio
async def test_embed_async():
    e = OpenAIEmbedding(model='text-embedding-ada-002')
    resp = await e.apredict('Hello, world!')

    assert len(resp) == e.shape[0]
    assert all(isinstance(x, float) for x in resp)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_batch_embed_async.yaml',
    filter_headers=['authorization'],
)
@pytest.mark.asyncio
async def test_batch_embed_async():
    e = OpenAIEmbedding(model='text-embedding-ada-002')
    resp = await e.apredict(['Hello', 'world!'], batch_size=1)

    assert len(resp) == 2
    assert all(len(x) == e.shape[0] for x in resp)
    assert all(isinstance(x, float) for y in resp for x in y)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_chat.yaml',
    filter_headers=['authorization'],
)
def test_chat():
    e = OpenAIChatCompletion(model='gpt-3.5-turbo', prompt='Hello, {context}')
    resp = e.predict('', one=True, context=['world!'])

    assert isinstance(resp, str)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_batch_chat.yaml',
    filter_headers=['authorization'],
)
def test_batch_chat():
    e = OpenAIChatCompletion(model='gpt-3.5-turbo')
    resp = e.predict(['Hello, world!'], one=False)

    assert isinstance(resp, list)
    assert isinstance(resp[0], str)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_chat_async.yaml',
    filter_headers=['authorization'],
)
@pytest.mark.asyncio
async def test_chat_async():
    e = OpenAIChatCompletion(model='gpt-3.5-turbo', prompt='Hello, {context}')
    resp = await e.apredict('', one=True, context=['world!'])

    assert isinstance(resp, str)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_batch_chat_async.yaml',
    filter_headers=['authorization'],
)
@pytest.mark.asyncio
async def test_batch_chat_async():
    e = OpenAIChatCompletion(model='gpt-3.5-turbo')
    resp = await e.apredict(['Hello, world!'], one=False)

    assert isinstance(resp, list)
    assert isinstance(resp[0], str)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_create_url.yaml',
    filter_headers=['authorization'],
    before_record_response=_record_only_png_signature_in_response,
)
def test_create_url():
    e = OpenAIImageCreation(
        model='dall-e', prompt='a close up, studio photographic portrait of a {context}'
    )
    resp = e.predict('', one=True, response_format='url', context=['cat'])

    # PNG 8-byte signature
    assert resp[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_create_url_batch.yaml',
    filter_headers=['authorization'],
    before_record_response=_record_only_png_signature_in_response,
)
def test_create_url_batch():
    e = OpenAIImageCreation(
        model='dall-e', prompt='a close up, studio photographic portrait of a'
    )
    resp = e.predict(['cat', 'dog'], response_format='url')

    for img in resp:
        # PNG 8-byte signature
        assert img[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_create_async.yaml',
    filter_headers=['authorization'],
    before_record_response=_record_only_png_signature_in_response,
)
@pytest.mark.asyncio
async def test_create_async():
    e = OpenAIImageCreation(
        model='dall-e', prompt='a close up, studio photographic portrait of a {context}'
    )
    resp = await e.apredict('', one=True, context=['cat'])

    assert isinstance(resp, bytes)

    # PNG 8-byte signature
    assert resp[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_create_url_async.yaml',
    filter_headers=['authorization'],
    before_record_response=_record_only_png_signature_in_response,
)
@pytest.mark.asyncio
async def test_create_url_async():
    e = OpenAIImageCreation(
        model='dall-e', prompt='a close up, studio photographic portrait of a {context}'
    )
    resp = await e.apredict('', one=True, response_format='url', context=['cat'])

    # PNG 8-byte signature
    assert resp[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_create_url_async_batch.yaml',
    filter_headers=['authorization'],
    before_record_response=_record_only_png_signature_in_response,
)
@pytest.mark.asyncio
async def test_create_url_async_batch():
    e = OpenAIImageCreation(
        model='dall-e', prompt='a close up, studio photographic portrait of a'
    )
    resp = await e.apredict(['cat', 'dog'], response_format='url')

    for img in resp:
        # PNG 8-byte signature
        assert img[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_edit_url.yaml',
    filter_headers=['authorization'],
    before_record_response=_record_only_png_signature_in_response,
)
def test_edit_url():
    e = OpenAIImageEdit(
        model='dall-e', prompt='A celebration party at the launch of {context}'
    )
    with open('test/material/data/rickroll.png', 'rb') as f:
        buffer = io.BytesIO(f.read())
    resp = e.predict(buffer, one=True, response_format='url', context=['superduperdb'])
    buffer.close()

    # PNG 8-byte signature
    assert resp[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_edit_url_batch.yaml',
    filter_headers=['authorization'],
    before_record_response=_record_only_png_signature_in_response,
)
def test_edit_url_batch():
    e = OpenAIImageEdit(
        model='dall-e', prompt='A celebration party at the launch of superduperdb'
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


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_edit_async.yaml',
    filter_headers=['authorization'],
    before_record_response=_record_only_png_signature_in_response,
)
@pytest.mark.asyncio
async def test_edit_async():
    e = OpenAIImageEdit(
        model='dall-e', prompt='A celebration party at the launch of {context}'
    )
    with open('test/material/data/rickroll.png', 'rb') as f:
        buffer = io.BytesIO(f.read())
    resp = await e.apredict(buffer, one=True, context=['superduperdb'])
    buffer.close()

    assert isinstance(resp, bytes)

    # PNG 8-byte signature
    assert resp[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_edit_url_async.yaml',
    filter_headers=['authorization'],
    before_record_response=_record_only_png_signature_in_response,
)
@pytest.mark.asyncio
async def test_edit_url_async():
    e = OpenAIImageEdit(
        model='dall-e', prompt='A celebration party at the launch of {context}'
    )
    with open('test/material/data/rickroll.png', 'rb') as f:
        buffer = io.BytesIO(f.read())
    resp = await e.apredict(
        buffer, one=True, response_format='url', context=['superduperdb']
    )
    buffer.close()

    # PNG 8-byte signature
    assert resp[0:16] == PNG_BYTE_SIGNATURE


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_edit_url_async_batch.yaml',
    filter_headers=['authorization'],
    before_record_response=_record_only_png_signature_in_response,
)
@pytest.mark.asyncio
async def test_edit_url_async_batch():
    e = OpenAIImageEdit(
        model='dall-e', prompt='A celebration party at the launch of superduperdb'
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


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_transcribe.yaml',
    filter_headers=['authorization'],
)
def test_transcribe():
    with open('test/material/data/test.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
    buffer.name = 'test.wav'
    prompt = (
        'i have some advice for you. write all text in lower-case.'
        'only make an exception for the following words: {context}'
    )
    e = OpenAIAudioTranscription(model='whisper-1', prompt=prompt)
    resp = e.predict(buffer, one=True, context=['United States'])
    buffer.close()

    assert 'United States' in resp


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_batch_transcribe.yaml',
    filter_headers=['authorization'],
)
def test_batch_transcribe():
    with open('test/material/data/test.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
        buffer.name = 'test.wav'

    with open('test/material/data/test.wav', 'rb') as f:
        buffer2 = io.BytesIO(f.read())
        buffer2.name = 'test.wav'

    e = OpenAIAudioTranscription(model='whisper-1')
    resp = e.predict([buffer, buffer2], one=False, batch_size=1)
    buffer.close()

    assert len(resp) == 2
    assert resp[0] == resp[1]
    assert 'United States' in resp[0]


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_transcribe_async.yaml',
    filter_headers=['authorization'],
)
@pytest.mark.asyncio
async def test_transcribe_async():
    with open('test/material/data/test.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
    buffer.name = 'test.wav'
    prompt = (
        'i have some advice for you. write all text in lower-case.'
        'only make an exception for the following words: {context}'
    )
    e = OpenAIAudioTranscription(model='whisper-1', prompt=prompt)
    resp = await e.apredict(buffer, one=True, context=['United States'])
    buffer.close()

    assert 'United States' in resp


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_batch_transcribe_async.yaml',
    filter_headers=['authorization'],
)
@pytest.mark.asyncio
async def test_batch_transcribe_async():
    with open('test/material/data/test.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
        buffer.name = 'test1.wav'

    with open('test/material/data/test.wav', 'rb') as f:
        buffer2 = io.BytesIO(f.read())
        buffer2.name = 'test1.wav'
    e = OpenAIAudioTranscription(model='whisper-1')

    resp = await e.apredict([buffer, buffer2], one=False, batch_size=1)
    buffer.close()

    assert len(resp) == 2
    assert resp[0] == resp[1]
    assert 'United States' in resp[0]


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_translate.yaml',
    filter_headers=['authorization'],
)
def test_translate():
    with open('test/material/data/german.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
    buffer.name = 'test.wav'
    prompt = (
        'i have some advice for you. write all text in lower-case.'
        'only make an exception for the following words: {context}'
    )
    e = OpenAIAudioTranslation(model='whisper-1', prompt=prompt)
    resp = e.predict(buffer, one=True, context=['Emmerich'])
    buffer.close()

    assert 'station' in resp


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_batch_translate.yaml',
    filter_headers=['authorization'],
)
def test_batch_translate():
    with open('test/material/data/german.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
        buffer.name = 'test.wav'

    with open('test/material/data/german.wav', 'rb') as f:
        buffer2 = io.BytesIO(f.read())
        buffer2.name = 'test.wav'

    e = OpenAIAudioTranslation(model='whisper-1')
    resp = e.predict([buffer, buffer2], one=False, batch_size=1)
    buffer.close()

    assert len(resp) == 2
    assert resp[0] == resp[1]
    assert 'station' in resp[0]


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_translate_async.yaml',
    filter_headers=['authorization'],
)
@pytest.mark.asyncio
async def test_translate_async():
    with open('test/material/data/german.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
    buffer.name = 'test.wav'
    prompt = (
        'i have some advice for you. write all text in lower-case.'
        'only make an exception for the following words: {context}'
    )
    e = OpenAIAudioTranslation(model='whisper-1', prompt=prompt)
    resp = await e.apredict(buffer, one=True, context=['Emmerich'])
    buffer.close()

    assert 'station' in resp


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_batch_translate_async.yaml',
    filter_headers=['authorization'],
)
@pytest.mark.asyncio
async def test_batch_translate_async():
    with open('test/material/data/german.wav', 'rb') as f:
        buffer = io.BytesIO(f.read())
        buffer.name = 'test1.wav'

    with open('test/material/data/german.wav', 'rb') as f:
        buffer2 = io.BytesIO(f.read())
        buffer2.name = 'test1.wav'
    e = OpenAIAudioTranslation(model='whisper-1')

    resp = await e.apredict([buffer, buffer2], one=False, batch_size=1)
    buffer.close()

    assert len(resp) == 2
    assert resp[0] == resp[1]
    assert 'station' in resp[0]
