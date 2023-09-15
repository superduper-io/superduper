import pytest
import vcr

from superduperdb.ext.cohere import CohereEmbed, CohereGenerate

CASSETTE_DIR = 'test/integration/ext/cohere/cassettes'


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_embed_one.yaml',
    filter_headers=['authorization'],
)
def test_embed_one():
    embed = CohereEmbed(model='embed-english-v2.0')
    resp = embed.predict('Hello world')

    assert len(resp) == embed.shape[0]
    assert isinstance(resp, list)
    assert all(isinstance(x, float) for x in resp)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_embed_batch.yaml',
    filter_headers=['authorization'],
)
def test_embed_batch():
    embed = CohereEmbed(model='embed-english-v2.0')
    resp = embed.predict(['Hello', 'world'], batch_size=1)

    assert len(resp) == 2
    assert len(resp[0]) == embed.shape[0]
    assert isinstance(resp[0], list)
    assert all(isinstance(x, float) for x in resp[0])


@pytest.mark.asyncio
@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_async_embed_one.yaml',
    filter_headers=['authorization'],
)
async def test_async_embed_one():
    embed = CohereEmbed(model='embed-english-v2.0')
    resp = await embed.apredict('Hello world')

    assert len(resp) == embed.shape[0]
    assert isinstance(resp, list)
    assert all(isinstance(x, float) for x in resp)


@pytest.mark.asyncio
@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_async_embed_batch.yaml',
    filter_headers=['authorization'],
)
async def test_async_embed_batch():
    embed = CohereEmbed(model='embed-english-v2.0')
    resp = await embed.apredict(['Hello', 'world'], batch_size=1)

    assert len(resp) == 2
    assert len(resp[0]) == embed.shape[0]
    assert isinstance(resp[0], list)
    assert all(isinstance(x, float) for x in resp[0])


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_generate.yaml',
    filter_headers=['authorization'],
)
def test_generate():
    e = CohereGenerate(model='base-light', prompt='Hello, {context}')
    resp = e.predict('', one=True, context=['world!'])

    assert isinstance(resp, str)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_batch_generate.yaml',
    filter_headers=['authorization'],
)
def test_batch_generate():
    e = CohereGenerate(model='base-light')
    resp = e.predict(['Hello, world!'], one=False)

    assert isinstance(resp, list)
    assert isinstance(resp[0], str)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_generate_async.yaml',
    filter_headers=['authorization'],
)
@pytest.mark.asyncio
async def test_chat_async():
    e = CohereGenerate(model='base-light', prompt='Hello, {context}')
    resp = await e.apredict('', one=True, context=['world!'])

    assert isinstance(resp, str)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_batch_chat_async.yaml',
    filter_headers=['authorization'],
)
@pytest.mark.asyncio
async def test_batch_chat_async():
    e = CohereGenerate(model='base-light')
    resp = await e.apredict(['Hello, world!'], one=False)

    assert isinstance(resp, list)
    assert isinstance(resp[0], str)
