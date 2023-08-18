import os
import random

import lorem
import pytest
from pymongo import MongoClient

import superduperdb as s
from superduperdb import superduper
from superduperdb.base.config import SelfHosted
from superduperdb.container.document import Document
from superduperdb.container.listener import Listener
from superduperdb.container.model import Model
from superduperdb.container.vector_index import VectorIndex
from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.vector.encoder import vector

ATLAS_VECTOR_URI = os.environ.get('ATLAS_VECTOR_URI')


def random_vector_model(x):
    return [random.random() for _ in range(16)]


@pytest.fixture()
def atlas_search_config():
    previous = s.CFG.vector_search.type
    s.CFG.vector_search.type = SelfHosted()
    yield
    s.CFG.vector_search.type = previous


@pytest.mark.skipif(ATLAS_VECTOR_URI is None, reason='Only atlas deployments relevant.')
def test_setup_atlas_vector_search(atlas_search_config):
    model = Model(
        identifier='test-model', object=random_vector_model, encoder=vector(shape=(16,))
    )
    client = MongoClient(ATLAS_VECTOR_URI)
    db = superduper(client.test_atlas_vector_search)
    collection = Collection('docs')

    vector_indexes = db.databackend.list_vector_indexes()

    assert not vector_indexes

    db.execute(
        collection.insert_many(
            [Document({'text': lorem.sentence()}) for _ in range(50)]
        )
    )
    db.add(
        VectorIndex(
            'test-vector-index',
            indexing_listener=Listener(
                model=model,
                key='text',
                select=collection.find(),
            ),
        )
    )

    assert 'test-vector-index' in db.show('vector_index')
    assert 'test-vector-index' in db.databackend.list_vector_indexes()


@pytest.mark.skipif(ATLAS_VECTOR_URI is None, reason='Only atlas deployments relevant.')
def test_use_atlas_vector_search(atlas_search_config):
    client = MongoClient(ATLAS_VECTOR_URI)
    db = superduper(client.test_atlas_vector_search)
    collection = Collection('docs')

    query = collection.like(
        Document({'text': 'This is a test'}), n=5, vector_index='test-vector-index'
    ).find()

    it = 0
    for r in db.execute(query):
        print(r)
        it += 1

    assert it == 4
