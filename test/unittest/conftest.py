import random

import pytest
import torch

from superduperdb.container.dataset import Dataset
from superduperdb.container.document import Document
from superduperdb.container.listener import Listener
from superduperdb.container.vector_index import VectorIndex
from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.numpy.array import array
from superduperdb.ext.pillow.image import pil_image
from superduperdb.ext.torch.model import TorchModel
from superduperdb.ext.torch.tensor import tensor


@pytest.fixture()
def database_with_float_tensors_32(empty_database):
    empty_database.add(tensor(torch.float, shape=(32,)))

    yield empty_database

    empty_database.remove('encoder', 'torch.float32[32]', force=True)


@pytest.fixture()
def database_with_float_tensors_16(empty_database):
    empty_database.add(tensor(torch.float, shape=(16,)))

    yield empty_database

    empty_database.remove('encoder', 'torch.float32[16]', force=True)


@pytest.fixture()
def database_with_float_tensors_8(empty_database):
    empty_database.add(tensor(torch.float, shape=(8,)))

    yield empty_database

    empty_database.remove('encoder', 'torch.float32[8]', force=True)


@pytest.fixture()
def database_with_array(empty_database):
    empty_database.add(array('float32', shape=(32,)))

    yield empty_database

    empty_database.remove('encoder', 'numpy.float32[32]', force=True)


@pytest.fixture()
def database_with_pil_image(empty_database):
    empty_database.add(pil_image)

    yield empty_database

    empty_database.remove('encoder', 'pil_image', force=True)


@pytest.fixture()
def database_with_random_tensor_data(database_with_float_tensors_32):
    float_tensor = database_with_float_tensors_32.encoders['torch.float32[32]']
    data = []
    for _ in range(250):
        x = torch.randn(32)
        y = int(random.random() > 0.5)
        z = torch.randn(32)
        data.append(
            Document(
                {
                    'x': float_tensor(x),
                    'y': y,
                    'z': float_tensor(z),
                }
            )
        )
    database_with_float_tensors_32.execute(
        Collection(name='documents').insert_many(data, refresh=False)
    )

    yield database_with_float_tensors_32

    database_with_float_tensors_32.execute(Collection(name='documents').delete_many({}))


@pytest.fixture()
def database_with_dataset(database_with_random_tensor_data):
    d = Dataset(
        identifier='my_valid',
        select=Collection(name='documents').find({'_fold': 'valid'}),
        sample_size=100,
    )
    database_with_random_tensor_data.add(d)

    yield database_with_random_tensor_data

    database_with_random_tensor_data.remove('dataset', 'my_valid', force=True)


@pytest.fixture()
# each fixture adds an encoder to the database, and both encoders are required
def database_with_torch_model_a(
    database_with_float_tensors_32, database_with_float_tensors_16
):
    database_with_float_tensors_32.add(
        TorchModel(
            object=torch.nn.Linear(32, 16),
            identifier='linear_a',
            encoder='torch.float32[16]',
        )
    )

    yield database_with_float_tensors_32

    database_with_float_tensors_32.remove('model', 'linear_a', force=True)


@pytest.fixture()
def database_with_listener_torch_model_a(database_with_torch_model_a):
    database_with_torch_model_a.distributed = False
    database_with_torch_model_a.add(
        Listener(
            model='linear_a',
            select=Collection(name='documents').find(),
            key='x',
        )
    )

    yield database_with_torch_model_a

    database_with_torch_model_a.remove('listener', 'linear_a/x', force=True)


@pytest.fixture()
# each fixture adds an encoder to the database, and all encoders are required
def database_with_torch_model_b(
    database_with_float_tensors_32,
    database_with_float_tensors_16,
    database_with_float_tensors_8,
):
    database_with_float_tensors_32.add(
        TorchModel(
            object=torch.nn.Linear(16, 8),
            identifier='linear_b',
            encoder='torch.float32[8]',
        ),
    )

    yield database_with_float_tensors_32

    database_with_float_tensors_32.remove('model', 'linear_b', force=True)


@pytest.fixture()
# database_with_torch_model_a required as linear_a model required in database
def database_with_vector_index(
    database_with_random_tensor_data, database_with_torch_model_a
):
    database_with_random_tensor_data.add(
        Listener(
            select=Collection(name='documents').find(),
            key='x',
            model='linear_a',
        )
    )
    database_with_random_tensor_data.add(
        Listener(
            select=Collection(name='documents').find(),
            key='z',
            model='linear_a',
        )
    )
    database_with_random_tensor_data.add(
        VectorIndex(
            identifier='test_vector_search',
            indexing_listener='linear_a/x',
            compatible_listener='linear_a/z',
        )
    )

    yield database_with_random_tensor_data

    database_with_random_tensor_data.remove(
        'vector_index', 'test_vector_search', force=True
    )
    database_with_random_tensor_data.remove('listener', 'linear_a/x', force=True)
    database_with_random_tensor_data.remove('listener', 'linear_a/z', force=True)


@pytest.fixture()
def a_single_document():
    float_tensor = tensor(torch.float, shape=(32,))
    x = torch.randn(32)
    y = int(random.random() > 0.5)
    z = torch.randn(32)
    return Document(
        {
            'x': float_tensor(x),
            'y': y,
            'z': float_tensor(z),
        }
    )


@pytest.fixture()
def multiple_documents():
    float_tensor = tensor(torch.float, shape=(32,))
    data = []
    for i in range(10):
        x = torch.randn(32)
        y = int(random.random() > 0.5)
        z = torch.randn(32)
        data.append(
            Document(
                {
                    'x': float_tensor(x),
                    'y': y,
                    'z': float_tensor(z),
                    'update': True,
                }
            )
        )

    return data
