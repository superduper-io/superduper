import pytest
import random
import time

from tests.fixtures.collection import (
    collection_hashes, test_document, test_document2, delete, special_test_document,
    collection_many_models, random_string, loaded_model, loaded_converter,
    cleanup_models, collection_no_hashes
)


def test_find_one(collection_hashes, test_document):
    r1 = collection_hashes.find_one({'$like': {'document': test_document, 'n': 10}})
    r2 = collection_hashes.find_one({
        '$like': {'document': test_document, 'n': 10},
        '_id': r1['_id']
    })
    assert r1['_id'] == r2['_id']


def test_find(collection_hashes, test_document):
    r1 = collection_hashes.find_one()
    r2 = collection_hashes.find({
        '$like': {'document': test_document, 'n': 10},
        '_id': r1['_id']
    })
    assert r1['_id'] == next(r2)['_id']


def test_insert_one(collection_hashes, special_test_document, delete):
    collection_hashes.insert_one(special_test_document)
    assert collection_hashes.count_documents({}) == 11
    docs = list(collection_hashes.find())
    assert all(['dummy' in x['_outputs']['_base'] for x in docs])


def test_insert_many(collection_hashes, test_document, test_document2, delete):
    collection_hashes.insert_many([test_document, test_document2])
    time.sleep(1)
    assert collection_hashes.count_documents({}) == 12
    docs = list(collection_hashes.find())
    assert all(['dummy' in x['_outputs']['_base'] for x in docs])


def test_insert_many_with_dependencies(collection_many_models):
    lookup = {True: 'apple', False: 'pear'}
    for i in range(10):
        if i < 8:
            collection_many_models.insert_one({'test': random_string(), '_fold': 'train',
                                               'fruit': lookup[random.random() < 0.5]})
        else:
            collection_many_models.insert_one({'test': random_string(), '_fold': 'valid',
                                               'fruit': lookup[random.random() < 0.5]})


def test_get_hash_set(collection_hashes):
    assert tuple(collection_hashes.hash_set.h.shape) == (10, 10)
    collection_hashes.semantic_index = 'valid_only'
    assert tuple(collection_hashes.hash_set.h.shape) == (2, 10)


def test_get_plan(collection_many_models):
    collection_many_models._create_plan()


def test_create_model(collection_no_hashes, loaded_model, loaded_converter, cleanup_models):
    collection_no_hashes.create_model(
        name='test-model',
        object=loaded_model,
        converter={'name': 'float_tensor', 'object': loaded_converter},
        active=False,
    )
    print(collection_no_hashes.models['test-model'])
    with pytest.raises(AssertionError):
        collection_no_hashes.create_model(
            name='test-model',
            object=loaded_model,
            converter={'name': 'float_tensor', 'object': loaded_converter},
            active=False
        )
