import json

import openai
import pytest
import vcr

from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.document import Document
from superduperdb.components.listener import Listener
from superduperdb.components.vector_index import VectorIndex
from superduperdb.ext.openai.model import OpenAIChatCompletion, OpenAIEmbedding

CASSETTE_DIR = 'test/unittest/ext/cassettes'


@pytest.fixture
def open_ai_with_rhymes(local_empty_db, monkeypatch):
    with open('test/material/data/rhymes.json') as f:
        data = json.load(f)
    for i, r in enumerate(data):
        data[i] = Document({'story': r.replace('\n', ' ')})
    monkeypatch.setattr(openai, 'api_key', 'sk-TopSecret')
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-TopSecret')
    local_empty_db.execute(Collection('openai').insert_many(data))
    yield local_empty_db
    local_empty_db.remove('model', 'gpt-3.5-turbo', force=True)
    local_empty_db.remove('model', 'text-embedding-ada-002', force=True)


# TODO: Mock OpenAI API instead of using VCR
@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_retrieve_with_similar_context.yaml',
    filter_headers=['authorization'],
)
def test_retrieve_with_similar_context(open_ai_with_rhymes):
    db = open_ai_with_rhymes
    m = OpenAIChatCompletion(
        model='gpt-3.5-turbo',
        prompt=(
            'Use the following facts to answer this question\n'
            '{context}\n\n'
            'Here\'s the question:\n'
        ),
    )
    db.add(m)
    vectorizer = OpenAIEmbedding('text-embedding-ada-002')
    db.add(
        VectorIndex(
            identifier='openai-index',
            indexing_listener=Listener(
                model=vectorizer,
                key='story',
                select=Collection('openai').find(),
            ),
        )
    )

    models = db.show('model')
    assert set(models) == {'text-embedding-ada-002', 'gpt-3.5-turbo'}
    r = db.execute(Collection('openai').find_one())
    assert '_outputs' in r.content

    input = 'Is covid a hoax?'

    prediction = db.predict(
        model_name='gpt-3.5-turbo',
        input=input,
        context_select=Collection('openai')
        .like({'story': input}, n=1, vector_index='openai-index')
        .find(),
        context_key='story',
    )

    assert isinstance(prediction[0], Document)
    assert 'hoax' in prediction[0].content

    assert isinstance(prediction[1][0], Document)
    assert isinstance(prediction[1][0].content, str)
