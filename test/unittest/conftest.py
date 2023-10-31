import random
import uuid
from pathlib import Path

import pytest

try:
    import torch

    from superduperdb.ext.torch.encoder import tensor
    from superduperdb.ext.torch.model import TorchModel
except ImportError:
    torch = None

from superduperdb import CFG
from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.build import build_datalayer
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.components.dataset import Dataset
from superduperdb.components.listener import Listener
from superduperdb.components.vector_index import VectorIndex
from superduperdb.ext.pillow.encoder import pil_image

n_data_points = 250

LOCAL_TEST_N_DATA_POINTS = 5

MONGOMOCK_URI = 'mongomock:///test_db'


@pytest.fixture
def valid_dataset():
    d = Dataset(
        identifier='my_valid',
        select=Collection('documents').find({'_fold': 'valid'}),
        sample_size=100,
    )
    return d


def add_random_data(
    data_layer: Datalayer,
    collection_name: str = 'documents',
    number_data_points: int = n_data_points,
):
    float_tensor = data_layer.encoders['torch.float32[32]']
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

    if data:
        data_layer.execute(
            Collection(collection_name).insert_many(data),
            refresh=False,
        )


def add_encoders(data_layer: Datalayer):
    for n in [8, 16, 32]:
        data_layer.add(tensor(torch.float, shape=(n,)))
    data_layer.add(pil_image)


def add_models(data_layer: Datalayer):
    # identifier, weight_shape, encoder
    params = [
        ['linear_a', (32, 16), 'torch.float32[16]'],
        ['linear_b', (16, 8), 'torch.float32[8]'],
    ]
    for identifier, weight_shape, encoder in params:
        data_layer.add(
            TorchModel(
                object=torch.nn.Linear(*weight_shape),
                identifier=identifier,
                encoder=encoder,
            )
        )


def add_vector_index(
    data_layer: Datalayer, collection_name='documents', identifier='test_vector_search'
):
    # TODO: Support configurable key and model
    data_layer.add(
        Listener(
            select=Collection(collection_name).find(),
            key='x',
            model='linear_a',
        )
    )
    data_layer.add(
        Listener(
            select=Collection(collection_name).find(),
            key='z',
            model='linear_a',
        )
    )
    vi = VectorIndex(
        identifier=identifier,
        indexing_listener='linear_a/x',
        compatible_listener='linear_a/z',
    )
    data_layer.add(vi)


@pytest.fixture(scope='session')
def image_url():
    path = Path(__file__).parent.parent / 'material' / 'data' / '1x1.png'
    return f'file://{path}'


@pytest.fixture(scope='session')
def global_identifier_of_vector_index() -> str:
    return 'global_identifier_of_vector_index'


def setup_data_layer(data_layer, **kwargs):
    # TODO: support more parameters to control the setup
    add_encoders(data_layer)
    n_data = kwargs.get('n_data', n_data_points)
    add_random_data(data_layer, number_data_points=n_data)
    if kwargs.get('add_models', True):
        add_models(data_layer)
    if kwargs.get('add_vector_index', True):
        add_vector_index(data_layer)


@pytest.fixture(scope='session')
def data_layer() -> Datalayer:
    _data_layer = build_datalayer(CFG, data_backend=MONGOMOCK_URI)
    setup_data_layer(_data_layer)
    return _data_layer


@pytest.fixture
def local_data_layer(request) -> Datalayer:
    _data_layer = build_datalayer(CFG, data_backend=MONGOMOCK_URI)
    setup_config = getattr(request, 'param', {'n_data': LOCAL_TEST_N_DATA_POINTS})
    setup_data_layer(_data_layer, **setup_config)
    return _data_layer


@pytest.fixture
def local_empty_data_layer(request) -> Datalayer:
    _data_layer = build_datalayer(CFG, data_backend=MONGOMOCK_URI)
    return _data_layer


@pytest.fixture
def empty_collection() -> Collection:
    return Collection(str(uuid.uuid4()))


@pytest.fixture
def local_collection_with_random_data(data_layer: Datalayer, request) -> Collection:
    collection_name = str(uuid.uuid4())
    number_data_points = getattr(request, 'param', n_data_points)
    add_random_data(data_layer, collection_name, number_data_points)
    yield Collection(collection_name)
