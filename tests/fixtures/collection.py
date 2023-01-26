import pytest
import torch

from superduperdb.client import the_client
from tests.material.types import FloatTensor, RawBytes
from tests.material.measures import css
from tests.material.models import ModelAttributes
from tests.material.models import NoForward


@pytest.fixture()
def random_vectors():
    the_client.drop_database('test_db')
    the_client.drop_database('_test_db:documents:files')
    coll = the_client.test_db.documents
    coll.create_type('float_tensor', FloatTensor())
    coll.create_type('raw_bytes', RawBytes())
    coll.create_measure('css', css)
    coll.create_model(
        'linear',
        torch.nn.Linear(32, 16),
        type='float_tensor',
        key='x',
        active=True,
    )
    coll.create_model(
        'model_attributes',
        ModelAttributes(32, 8, 4),
        active=False,
        type='float_tensor',
    )
    coll.create_model(
        'model_attributes.linear1',
        active=True,
        type='float_tensor',
        key='x',
    )
    coll.create_model(
        'model_attributes.linear2',
        active=True,
        type='float_tensor',
        key='x',
    )
    coll.create_model(
        'other_linear',
        torch.nn.Linear(16, 8),
        type='float_tensor',
        key='x',
        active=True,
        features={'x': 'linear'},
    )
    coll['_meta'].insert_one({'key': 'semantic_index', 'value': 'linear'})
    data = []
    eps = 0.01
    N = 500
    for i in range(N * 2):
        if i < N:
            x = torch.randn(32)
            y = torch.randn(32) * eps + x
            label = 0
        else:
            x = torch.randn(32) + 1
            y = torch.randn(32) * eps + x
            label = 1
        data.append({'i': i, 'x': x, 'y': y, 'label': label})
    coll.insert_many(data)
    coll.create_model(
        'no_forward',
        NoForward(),
        type='float_tensor',
        key='x',
        active=True,
        loader_kwargs={'num_workers': 2},
    )
    coll.create_validation_set('valid', {})
    yield coll
    the_client.drop_database('test_db')
    the_client.drop_database('_test_db:documents:files')


@pytest.fixture()
def empty():
    yield the_client.test_db.documents
    the_client.drop_database('test_db')
    the_client.drop_database('_test_db:documents:files')


@pytest.fixture(params=[0, 2])
def with_urls(request):
    the_client.drop_database('test_db')
    the_client.drop_database('_test_db:documents:files')
    the_client.test_db.documents.create_type('raw_bytes', RawBytes())
    collection = the_client.test_db.documents
    collection['_meta'].insert_one({'key': 'n_download_workers', 'value': request.param})
    docs = [
        {
            'item': {
                '_content': {
                    'url': 'https://www.superduperdb.com/logos/white.png',
                    'type': 'raw_bytes',
                }
            },
            'other': {
                'item': {
                    '_content': {
                        'url': 'https://www.superduperdb.com/logos/white.png',
                        'type': 'raw_bytes',
                    }
                }
            }
        }
        for _ in range(2)
    ]
    collection.insert_many(docs)
    yield collection
    the_client.drop_database('test_db')
    the_client.drop_database('_test_db:documents:files')