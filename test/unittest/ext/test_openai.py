import json
import os
from test.db_config import DBConfig

import openai
import pytest
import vcr

from superduperdb.backends.ibis.field_types import dtype
from superduperdb.backends.ibis.query import Schema, Table
from superduperdb.backends.mongodb.data_backend import MongoDataBackend
from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.document import Document
from superduperdb.components.listener import Listener
from superduperdb.components.vector_index import VectorIndex
from superduperdb.ext.openai.model import (
    OpenAIChatCompletion,
    OpenAIEmbedding,
    _available_models,
)

CASSETTE_DIR = 'test/unittest/ext/cassettes'


@pytest.fixture(autouse=True)
def mock_lru_cache():
    _available_models.cache_clear()


@pytest.fixture
def open_ai_with_rhymes(db, monkeypatch):
    with open('test/material/data/rhymes.json') as f:
        data = json.load(f)[:10]

    if os.getenv('OPENAI_API_KEY') is None:
        monkeypatch.setattr(openai, 'api_key', 'sk-TopSecret')
        monkeypatch.setenv('OPENAI_API_KEY', 'sk-TopSecret')
    if isinstance(db.databackend, MongoDataBackend):
        for i, r in enumerate(data):
            data[i] = Document({'story': r.replace('\n', ' ')})
        db.execute(Collection('openai').insert_many(data))
    else:
        for i, r in enumerate(data):
            data[i] = Document({'story': r.replace('\n', ' '), 'id': str(i)})
        schema = Schema(
            identifier='my_table',
            fields={
                'id': dtype('str'),
                'story': dtype('str'),
            },
        )
        t = Table(identifier='my_table', schema=schema)
        db.add(t)
        insert = t.insert(data)
        db.execute(insert)
    yield db
    db.remove('model', 'gpt-3.5-turbo', force=True)
    db.remove('model', 'text-embedding-ada-002', force=True)


@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_retrieve_with_similar_context.yaml',
    filter_headers=['authorization'],
    record_on_exception=False,
    ignore_localhost=True,
)
@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_retrieve_with_similar_context(open_ai_with_rhymes):
    db = open_ai_with_rhymes
    m = OpenAIChatCompletion(
        identifier='gpt-3.5-turbo',
        prompt=(
            'Use the following facts to answer this question\n'
            '{context}\n\n'
            'Here\'s the question:\n'
        ),
    )
    db.add(m)
    vectorizer = OpenAIEmbedding('text-embedding-ada-002')

    if isinstance(db.databackend, MongoDataBackend):
        select = Collection('openai').find({})
    else:
        t = db.load('table', 'my_table')
        select = t.select('id', 'story')

    db.add(
        VectorIndex(
            identifier='openai-index',
            indexing_listener=Listener(
                model=vectorizer,
                key='story',
                select=select,
            ),
        )
    )

    models = db.show('model')
    assert set(models) == {'text-embedding-ada-002', 'gpt-3.5-turbo'}
    if isinstance(db.databackend, MongoDataBackend):
        r = list(db.execute(select))
        assert '_outputs' in r[0]
    else:
        r = db.execute(t.outputs(story='text-embedding-ada-002'))[0]
        assert '_outputs.story.text-embedding-ada-002.0' in r

    input = 'Is covid a hoax?'
    if isinstance(db.databackend, MongoDataBackend):
        context_select = (
            Collection('openai')
            .like({'story': input}, n=1, vector_index='openai-index')
            .find()
        )
    else:
        context_select = t.like(
            {'story': input}, n=1, vector_index='openai-index'
        ).select('id', 'story')

    prediction = db.predict(
        model_name='gpt-3.5-turbo',
        input=input,
        context_select=context_select,
        context_key='story',
    )

    assert isinstance(prediction[0], Document)
    assert 'hoax' in prediction[0].unpack()

    assert isinstance(prediction[1][0], Document)
