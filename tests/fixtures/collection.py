import pymongo
import pytest
import random
import torch

from sddb.models.converters import FloatTensor
import sddb.client


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


@pytest.fixture()
def collection_no_hashes():
    collection = pymongo.MongoClient().test_db.test_collection
    lookup = {True: 'apple', False: 'pear'}
    for i in range(10):
        if i < 8:
            collection.insert_one({'test': random_string(), '_fold': 'train',
                                   'fruit': lookup[random.random() < 0.5]})
        else:
            collection.insert_one({'test': random_string(), '_fold': 'valid',
                                   'fruit': lookup[random.random() < 0.5]})
    yield sddb.client.SddbClient().test_db.test_collection
    pymongo.MongoClient().drop_database('test_db')


@pytest.fixture()
def collection_hashes():
    collection = sddb.client.SddbClient().test_db.test_collection

    for i in range(10):
        if i < 8:
            collection.insert_one({
                'test': random_string(),
                '_outputs': {
                    '_base': {
                        'dummy': {
                            '_content': {
                                'bytes': FloatTensor.encode(torch.randn(10)),
                                'converter': 'sddb.converters.FloatTensor',
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
                                'bytes': FloatTensor.encode(torch.randn(10)),
                                'converter': 'sddb.converters.FloatTensor',
                            }
                        },
                        'valid_only': {
                            '_content': {
                                'bytes': FloatTensor.encode(torch.randn(10)),
                                'converter': 'sddb.converters.FloatTensor',
                            }
                        }
                    }
                },
                '_fold': 'valid',
            })

    collection.create_semantic_index({
        'name': 'dummy',
        'keys': ['_base'],
        'models': {
            '_base': {
                'type': 'import',
                'args': {
                    'path': 'tests.material.models.Dummy',
                    'kwargs': {},
                },
                'name': 'dummy',
                'converter': 'sddb.converters.FloatTensor',
            }
        },
        'target': '_base'
    })

    collection.create_semantic_index({
        'name': 'valid_only',
        'keys': ['_base'],
        'models': {
            '_base': {
                'type': 'import',
                'args': {
                    'path': 'tests.material.models.Dummy',
                    'kwargs': {},
                },
                'name': 'valid_only',
                'converter': 'sddb.converters.FloatTensor',
                'filter': {'_fold': 'valid'}
            }
        },
        'target': '_base'
    })

    collection = sddb.client.SddbClient().test_db.test_collection

    yield collection

    pymongo.MongoClient().drop_database('test_db')


@pytest.fixture()
def test_document():
    return {'test': 'abcd efgh'}


@pytest.fixture()
def special_test_document():
    return {'test': 'abcd efgh', 'special': True}


@pytest.fixture()
def test_document2():
    return {'test': 'efgh ijkl'}


@pytest.fixture()
def delete():
    yield
    pymongo.MongoClient().test_db.test_collection.delete_many({
        'test': {'$in': ['abcd efgh', 'efgh ijkl']}
    })
