import pytest
import vcr

from superduperdb.ext.openai.model import OpenAIChatCompletion, OpenAIEmbedding

CASSETTE_DIR = 'test/integration/ext/openai/cassettes'


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
