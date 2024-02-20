import pytest

try:
    import torch

    from superduperdb.ext.torch.model import TorchModel, TorchTrainerConfiguration
except ImportError:
    torch = None

from test.db_config import DBConfig

from superduperdb.backends.mongodb.data_backend import MongoDataBackend
from superduperdb.backends.mongodb.query import Collection
from superduperdb.components.datatype import DataType
from superduperdb.components.metric import Metric


class ToDict:
    def __init__(self):
        self.dict = dict(zip(list('abcdefghiklmnopqrstuvwyz'), range(26)))

    def __call__(self, input: str):
        return [self.dict[k] for k in input]


class TensorLookup:
    def __init__(self):
        self.d = torch.randn(26, 32)

    def __call__(self, x):
        return torch.stack([self.d[y] for y in x])


def pad_to_ten(x):
    to_stack = []
    for i, y in enumerate(x):
        out = torch.zeros(10, 32)
        y = y[:10]
        out[: y.shape[0], :] = y
        to_stack.append(out)
    return torch.stack(to_stack)


def my_loss(X, y):
    return torch.nn.functional.binary_cross_entropy_with_logits(
        X[:, 0], y.type(torch.float)
    )


def acc(x, y):
    return x == y


@pytest.fixture
def model():
    return TorchModel(
        object=torch.nn.Linear(32, 1),
        identifier='test',
        training_configuration=TorchTrainerConfiguration(
            identifier='my_configuration',
            objective=my_loss,
            loader_kwargs={'batch_size': 10},
            max_iterations=100,
            validation_interval=10,
        ),
        preferred_devices=('cpu',),
        postprocess=lambda x: int(torch.sigmoid(x).item() > 0.5),
        datatype=DataType(identifier='base'),
    )


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    'db',
    [
        (DBConfig.mongodb_data, {'n_data': 500}),
        (DBConfig.sqldb_data, {'n_data': 500}),
    ],
    indirect=True,
)
def test_fit(db, valid_dataset, model):
    db.add(valid_dataset)

    m = model

    if isinstance(db.databackend, MongoDataBackend):
        select = Collection('documents').find()
    else:
        table = db.load('table', 'documents')
        select = table.select('id', 'x', 'y', 'z', '_fold')

    m.fit(
        X='x',
        y='y',
        db=db,
        select=select,
        metrics=[Metric(identifier='acc', object=acc)],
        validation_sets=['my_valid'],
    )


def test_predict():
    # Check that the pre-process etc. has been called
    ...
