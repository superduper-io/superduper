import json
import random
from pathlib import Path
from test.material.metrics import PatK
from typing import Iterator

import lorem
import numpy
import pytest

try:
    from test.material.models import BinaryClassifier

    import torch

    from superduperdb.ext.torch.encoder import tensor
    from superduperdb.ext.torch.model import TorchModel
except ImportError:
    torch = None

from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.components.dataset import Dataset
from superduperdb.components.listener import Listener
from superduperdb.components.metric import Metric
from superduperdb.components.vector_index import VectorIndex
from superduperdb.ext.numpy.encoder import array
from superduperdb.ext.pillow.encoder import pil_image

n_data_points = 250


@pytest.fixture
def empty() -> Iterator[Datalayer]:
    from superduperdb import CFG
    from superduperdb.base.build import build_datalayer

    db = build_datalayer(CFG, data_backend='mongomock:///test_db')
    db.databackend.conn.is_mongos
    yield db
    db.databackend.conn.close()


@pytest.fixture()
def metric(empty):
    empty.add(Metric(identifier='p@1', object=PatK(1)))
    yield
    empty.remove('metric', 'p@1', force=True)


@pytest.fixture
def random_data_factory(float_tensors_32):
    float_tensor = float_tensors_32.encoders['torch.float32[32]']

    def _factory(number_data_points=n_data_points):
        data = []
        for i in range(number_data_points):
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
        float_tensors_32.execute(
            Collection('documents').insert_many(data),
            refresh=False,
        )
        return float_tensors_32

    return _factory


@pytest.fixture()
def random_data(random_data_factory):
    return random_data_factory()


@pytest.fixture()
def random_arrays(arrays):
    float_array = arrays.encoders['numpy.float32[32]']
    data = []
    for i in range(n_data_points):
        x = numpy.random.randn(32).astype(numpy.float32)
        y = int(random.random() > 0.5)
        data.append(Document({'x': float_array(x), 'y': y}))

    arrays.execute(Collection('documents').insert_many(data, refresh=False))
    yield arrays
    arrays.execute(Collection('documents').delete_many({}))


@pytest.fixture()
def a_single_insert(float_tensors_32):
    float_tensor = float_tensors_32.encoders['torch.float32[32]']
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
def an_insert(float_tensors_32):
    float_tensor = float_tensors_32.encoders['torch.float32[32]']
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
                    'update': False,
                }
            )
        )
    return data


@pytest.fixture()
def an_update(float_tensors_32):
    float_tensor = float_tensors_32.encoders['torch.float32[32]']
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


@pytest.fixture
def vector_index_factory(a_model):
    def _factory(db, identifier, **kwargs) -> VectorIndex:
        db.add(
            Listener(
                select=Collection('documents').find(),
                key='x',
                model='linear_a',
            )
        )
        db.add(
            Listener(
                select=Collection('documents').find(),
                key='z',
                model='linear_a',
            )
        )
        vi = VectorIndex(
            identifier=identifier,
            indexing_listener='linear_a/x',
            compatible_listener='linear_a/z',
            **kwargs,
        )
        db.add(vi)
        return vi

    return _factory


@pytest.fixture()
def with_vector_index(random_data, vector_index_factory):
    vector_index_factory(random_data, 'test_vector_search')
    yield random_data


@pytest.fixture()
def si_validation(random_data):
    d = Dataset(
        identifier='my_valid',
        select=Collection('documents').find({'_fold': 'valid'}),
        sample_size=100,
    )
    random_data.add(d)
    yield random_data


@pytest.fixture()
def float_tensors_32(empty):
    empty.add(tensor(torch.float, shape=(32,)))
    yield empty
    empty.remove('encoder', 'torch.float32[32]', force=True)


@pytest.fixture()
def float_tensors_16(empty):
    empty.add(tensor(torch.float, shape=(16,)))
    yield empty
    empty.remove('encoder', 'torch.float32[16]', force=True)


@pytest.fixture()
def float_tensors_8(empty):
    empty.add(tensor(torch.float, shape=(8,)))
    yield empty
    empty.remove('encoder', 'torch.float32[8]', force=True)


@pytest.fixture()
def arrays(empty):
    empty.add(array('float32', shape=(32,)))
    yield empty
    empty.remove('encoder', 'numpy.float32[32]', force=True)


@pytest.fixture()
def sentences(empty):
    data = []
    for _ in range(100):
        data.append(Document({'text': lorem.sentence()}))
    empty.execute(Collection('documents').insert_many(data))
    yield empty


@pytest.fixture()
def nursery_rhymes(empty):
    with open('test/material/data/rhymes.json') as f:
        data = json.load(f)
    for i in range(len(data)):
        data[i] = Document({'text': data[i]})
    empty.execute(Collection('documents').insert_many(data))
    yield empty


@pytest.fixture()
def image_type(empty):
    empty.add(pil_image)
    yield empty
    empty.remove('encoder', 'pil_image', force=True)


@pytest.fixture()
def a_model(float_tensors_32, float_tensors_16):
    float_tensors_32.add(
        TorchModel(
            object=torch.nn.Linear(32, 16),
            identifier='linear_a',
            encoder='torch.float32[16]',
        )
    )
    yield float_tensors_32
    try:
        float_tensors_32.remove('model', 'linear_a', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def a_model_base(float_tensors_32, float_tensors_16):
    float_tensors_32.add(
        TorchModel(
            object=torch.nn.Linear(32, 16),
            identifier='linear_a_base',
            encoder='torch.float32[16]',
            preprocess=lambda r: r['x'],
        ),
    )
    yield float_tensors_32
    try:
        float_tensors_32.remove('model', 'linear_a_base', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def a_listener(a_model):
    a_model.distributed = False
    a_model.add(
        Listener(
            model='linear_a',
            select=Collection('documents').find(),
            key='x',
        )
    )
    yield a_model
    a_model.remove('listener', 'linear_a/x', force=True)


@pytest.fixture()
def a_listener_base(a_model_base):
    a_model_base.add(
        Listener(
            model='linear_a_base',
            select=Collection('documents').find({}, {'_id': 0}),
            key='_base',
        )
    )
    yield a_model_base
    a_model_base.remove('listener', 'linear_a_base/_base', force=True)


@pytest.fixture()
def a_classifier(float_tensors_32):
    float_tensors_32.add(
        TorchModel(
            object=BinaryClassifier(32),
            identifier='classifier',
        ),
    )
    yield float_tensors_32
    try:
        float_tensors_32.remove('model', 'classifier', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def b_model(float_tensors_32, float_tensors_16, float_tensors_8):
    float_tensors_32.add(
        TorchModel(
            object=torch.nn.Linear(16, 8),
            identifier='linear_b',
            encoder='torch.float32[8]',
        ),
    )
    yield float_tensors_32
    try:
        float_tensors_32.remove('model', 'linear_b', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture()
def c_model(float_tensors_32, float_tensors_16):
    float_tensors_32.add(
        TorchModel(
            object=torch.nn.Linear(32, 16),
            identifier='linear_c',
            encoder='torch.float32[16]',
        ),
    )
    yield float_tensors_32
    try:
        float_tensors_32.remove('model', 'linear_c', force=True)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            return
        raise e


@pytest.fixture(scope='session')
def image_url():
    path = Path(__file__).parent.parent / 'material' / 'data' / '1x1.png'
    return f'file://{path}'
