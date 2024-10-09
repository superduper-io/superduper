import random
from typing import Iterator

import pytest
import torch
from superduper import superduper
from superduper.base.datalayer import Datalayer
from superduper.components.datatype import DataType

from superduper_torch.model import TorchModel
from superduper_torch.training import TorchTrainer


@pytest.fixture
def db() -> Iterator[Datalayer]:
    db = superduper("mongomock://test_db")

    yield db
    db.drop(force=True, data=True)


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
        preferred_devices=('cpu',),
        postprocess=lambda x: int(torch.sigmoid(x).item() > 0.5),
        datatype=DataType(identifier='base'),
    )


# TODO: The training task is not executed, but it was not tested.
@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_fit(db, model, capfd):

    data = []
    for i in range(500):
        x = torch.rand(32)
        y = int(random.random() > 0.5)
        z = torch.rand(32)
        fold = int(random.random() > 0.5)
        fold = "valid" if fold else "train"
        data.append({"id": str(i), "x": x, "y": y, "z": z, "_fold": fold})

    db.cfg.auto_schema = True
    db['documents'].insert(data).execute()

    select = db['documents'].select()
    trainer = TorchTrainer(
        key=('x', 'y'),
        select=select,
        identifier='my_trainer',
        objective=my_loss,
        loader_kwargs={'batch_size': 10},
        max_iterations=100,
        validation_interval=10,
    )

    model.trainer = trainer
    db.apply(model)

    captured = capfd.readouterr()

    # check that the training actually happened
    assert 'TRAIN; iteration:' in captured.out
