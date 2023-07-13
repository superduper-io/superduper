import pytest
import torch

from superduperdb.client import the_client
from tests.material.converters import FloatTensor


@pytest.fixture()
def random_vectors():
    the_client.drop_database('test_db')
    the_client.drop_database('_test_db:documents:files')
    coll = the_client.test_db.documents
    coll.create_converter('float_tensor', FloatTensor())
    coll.create_model(
        'linear',
        torch.nn.Linear(32, 16),
        converter='float_tensor',
        key='x',
    )
    data = []
    eps = 0.01
    for i in range(2000):
        x = torch.randn(32)
        y = torch.randn(32) * eps + x
        data.append({
            'i': i,
            'x': {
                '_content': {
                    'bytes': FloatTensor.encode(x),
                    'converter': 'float_tensor',
                }
            },
            'y': {
                '_content': {
                    'bytes': FloatTensor.encode(y),
                    'converter': 'float_tensor',
                }
            },
        })
    coll.insert_many(data)
    yield coll
    the_client.drop_database('test_db')
    the_client.drop_database('_test_db:documents:files')


@pytest.fixture()
def empty():
    yield the_client.test_db.documents
    the_client.drop_database('test_db')
    the_client.drop_database('_test_db:documents:files')

