import json
import os

import pytest

from superduperdb.base.document import Document
from superduperdb.base.vector_index import VectorIndex
from superduperdb.base.watcher import Watcher
from superduperdb.datalayer.mongodb.query import Collection
from superduperdb.ext.openai.model import OpenAIChatCompletion, OpenAIEmbedding

SKIP_PAID = os.environ.get('OPENAI_API_KEY') is None


@pytest.mark.skipif(SKIP_PAID, reason='OPENAI_API_KEY not set')
def test_open_ai_chat_completion():
    m = OpenAIChatCompletion('gpt-3.5-turbo')

    result = m.predict('Tell me about yourself?', one=True)

    print(result)

    assert 'openai' in result.lower()

    context = ['Imagine you are a woman from Dallas.', 'You like to ride horses.']

    m = OpenAIChatCompletion(
        model='gpt-3.5-turbo',
        prompt=(
            'Use the following facts to answer this question\n'
            '{context}\n\n'
            'Here\'s the question:\n'
        ),
    )
    result = m.predict(
        'Tell me about yourself?',
        context=context,
        one=True,
    )

    assert 'horse' in result.lower()
    assert 'dallas' in result.lower()

    print(result)


@pytest.fixture
def open_ai_with_rhymes(empty):
    with open('tests/material/data/rhymes.json') as f:
        data = json.load(f)
    for i, r in enumerate(data):
        data[i] = Document({'story': r.replace('\n', ' ')})
    empty.execute(Collection('openai').insert_many(data))
    yield empty
    empty.remove('model', 'gpt-3.5-turbo', force=True)
    empty.remove('model', 'text-embedding-ada-002', force=True)


@pytest.mark.skipif(SKIP_PAID, reason='OPENAI_API_KEY not set')
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
            indexing_watcher=Watcher(
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
        model='gpt-3.5-turbo',
        input=input,
        context_select=Collection('openai')
        .like({'story': input}, n=1, vector_index='openai-index')
        .find(),
        context_key='story',
    )

    print('PREDICTION IS:\n')
    print(prediction[0])

    print('\nCONTEXT IS:\n')
    print(prediction[1])
