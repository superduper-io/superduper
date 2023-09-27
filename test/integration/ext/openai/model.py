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
)

CASSETTE_DIR = 'test/integration/ext/openai/cassettes'

if os.getenv('OPENAI_API_KEY') is None:
    mp = pytest.MonkeyPatch()
    mp.setattr(openai, 'api_key', 'sk-TopSecret')
    mp.setenv('OPENAI_API_KEY', 'sk-TopSecret')


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
