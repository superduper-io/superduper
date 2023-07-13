import pytest
import torch

from superduperdb.client import the_client
from tests.material.converters import FloatTensor, RawBytes
from tests.material.measures import css
from tests.material.models import ModelAttributes
from tests.material.models import NoForward


@pytest.fixture()
def random_vectors():
    the_client.drop_database('test_db')
    the_client.drop_database('_test_db:documents:files')
    coll = the_client.test_db.documents
    coll.create_converter('float_tensor', FloatTensor())
    coll.create_converter('raw_bytes', RawBytes())
    coll.create_measure('css', css)
    coll.create_model(
        'linear',
        torch.nn.Linear(32, 16),
        converter='float_tensor',
        key='x',
        semantic_index=True,
        active=True,
        measure='css',
    )
    coll.create_model(
        'model_attributes',
        ModelAttributes(32, 8, 4),
        active=False,
    )
    coll.create_model(
        'model_attributes.linear1',
        active=True,
        converter='float_tensor',
        key='x',
    )
    coll.create_model(
        'model_attributes.linear2',
        active=True,
        converter='float_tensor',
        key='x',
    )
    coll.create_model(
        'other_linear',
        torch.nn.Linear(16, 8),
        converter='float_tensor',
        key='x',
        active=True,
        dependencies=('linear',),
        features={'x': 'linear'},
        requires='x',
        semantic_index=True,
        measure='css',
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
            y = torch.randn(32)  * eps + x
            label = 1
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
            'label': label,
        })
    coll.insert_many(data)
    coll.create_model(
        'no_forward',
        NoForward(),
        converter='float_tensor',
        key='x',
        active=True,
        requires='x',
        loader_kwargs={'num_workers': 2},
    )
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
    the_client.test_db.documents.create_converter('raw_bytes', RawBytes())
    collection = the_client.test_db.documents
    collection['_meta'].insert_one({'key': 'n_download_workers', 'value': request.param})
    docs = [
        {
            'item': {
                '_content': {
                    'url': 'https://www.superduperdb.com/logos/white.png',
                    'converter': 'raw_bytes',
                }
            },
            'other': {
                'item': {
                    '_content': {
                        'url': 'https://www.superduperdb.com/logos/white.png',
                        'converter': 'raw_bytes',
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