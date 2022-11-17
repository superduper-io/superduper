import pickle

import pymongo
import pytest
import random
import torch

import sddb.client
from sddb import cf


def random_word():
    n = random.randint(3, 10)
    letters = list('abcdefghijklmnopqrstuvwxyz')
    return ''.join([random.choice(letters) for _ in range(n)])


def random_string():
    n = random.randint(2, 7)
    str_ = ''
    for m in range(n):
        str_ += (random_word() + ' ')
    return str_.strip()


def mongo_client():
    return pymongo.MongoClient(**cf['mongodb'])


def sddb_client():
    return sddb.client.SddbClient(**cf['mongodb'])


@pytest.fixture
def collection_no_hashes():
    mongo_client().drop_database('test_db')
    collection = sddb_client().test_db.test_collection
    lookup = {True: 'apple', False: 'pear'}
    for i in range(10):
        if i < 8:
            collection.insert_one({'test': random_string(), '_fold': 'train',
                                   'fruit': lookup[random.random() < 0.5]})
        else:
            collection.insert_one({'test': random_string(), '_fold': 'valid',
                                   'fruit': lookup[random.random() < 0.5]})
    yield sddb_client().test_db.test_collection
    mongo_client().drop_database('test_db')


@pytest.fixture
def collection_many_models():
    with open('tests/material/pickles/dummy.pkl', 'rb') as f:
        m = pickle.load(f)
    collection = sddb_client().test_db.test_collection
    collection.create_model(
        object=m,
        name='extra-1',
        converter='float_tensor',
    )
    collection.create_model(
        object=m,
        name='extra-1-1',
        converter='float_tensor',
        dependencies=['extra-1'],
    )
    collection.create_model(
        object=m,
        name='extra-1-2',
        converter='float_tensor',
        dependencies=['extra-1'],
    )
    collection.create_model(
        object=m,
        name='extra-1-1-1',
        converter='float_tensor',
        dependencies=['extra-1-1'],
    )
    yield collection
    mongo_client().drop_database('test_db')


@pytest.fixture
def collection_hashes():
    sddb_client().drop_database('test_db')
    collection = sddb_client().test_db.test_collection
    with open('tests/material/fixtuers/float_tensor.pkl', 'rb') as f:
        float_tensor = pickle.load(f)
    collection.create_converter('float_tensor', float_tensor)

    for i in range(10):
        if i < 8:
            collection.insert_one({
                'test': random_string(),
                '_outputs': {
                    '_base': {
                        'dummy': {
                            '_content': {
                                'bytes': float_tensor.encode(torch.randn(10)),
                                'converter': 'float_tensor',
                            }
                        }
                    }
                },
                '_fold': 'train',
            })
        else:
            collection.insert_one({
                'test': random_string(),
                '_outputs': {
                    '_base': {
                        'dummy': {
                            '_content': {
                                'bytes': float_tensor.encode(torch.randn(10)),
                                'converter': 'float_tensor',
                            }
                        },
                        'valid_only': {
                            '_content': {
                                'bytes': float_tensor.encode(torch.randn(10)),
                                'converter': 'float_tensor',
                            }
                        }
                    }
                },
                '_fold': 'valid',
            })

    with open('tests/material/pickles/float_tensor.pkl') as f:
        dummy = pickle.load(f)

    collection.create_semantic_index(
        name='dummy',
        models=[
            {
                'object': dummy,
                'name': 'dummy',
                'converter': 'float_tensor',
                'active': True,
            }
        ],
    )

    collection.create_semantic_index(
        name='valid_only',
        models=[
            {
                'object': dummy,
                'name': 'valid_only',
                'converter': 'float_tensor',
                'filter': {'_fold': 'valid'},
                'key': '_base',
                'active': True,
            }
        ],
    )

    collection = sddb_client().test_db.test_collection

    yield collection

    mongo_client().drop_database('test_db')
    mongo_client().drop_database('_test_db:test_collection:files')


@pytest.fixture
def test_document():
    return {'test': 'abcd efgh'}


@pytest.fixture
def special_test_document():
    return {'test': 'abcd efgh', 'special': True}


@pytest.fixture
def test_document2():
    return {'test': 'efgh ijkl'}


@pytest.fixture
def delete():
    yield
    mongo_client().test_db.test_collection.delete_many({
        'test': {'$in': ['abcd efgh', 'efgh ijkl']}
    })


@pytest.fixture
def loaded_model():
    with open('tests/material/pickles/dummy.pkl', 'rb') as f:
        m = pickle.load(f)
        yield m


@pytest.fixture
def loaded_converter():
    with open('tests/material/pickles/float_tensor.pkl', 'rb') as f:
        m = pickle.load(f)
        yield m


@pytest.fixture
def cleanup_models():
    yield
    collection = sddb_client().test_db.test_collection
    collection.delete_models(['test-model'], force=True)
