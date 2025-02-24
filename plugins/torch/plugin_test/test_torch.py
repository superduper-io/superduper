import random
from typing import Iterator

import numpy as np
import pytest
import torch
from superduper import superduper
from superduper.base.datalayer import Datalayer
from superduper.base.datatype import pickle_encoder
from superduper.components.table import Table

from superduper_torch.model import TorchModel
from superduper_torch.training import TorchTrainer

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


@pytest.fixture
def db() -> Iterator[Datalayer]:
    db = superduper("mongomock://test_db", force_apply=True)

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
        datatype=pickle_encoder,
    )


def test_fit(db, model):

    db.apply(
        Table(
            'documents',
            fields={
                'id': 'str',
                'x': 'superduper_torch.Tensor[float32:32]',
                'y': 'int',
                'z': 'superduper_torch.Tensor[float32:32]',
                '_fold': 'str',
            },
        )
    )

    data = []
    for i in range(500):
        y = int(random.random() > 0.5)
        x = torch.rand(32) + y
        z = torch.rand(32)
        fold = int(random.random() > 0.5)
        fold = "valid" if fold else "train"
        data.append({"id": str(i), "x": x, "y": y, "z": z, "_fold": fold})

    db['documents'].insert(data)

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

    model = db.load("TorchModel", model.identifier)
    objective = model.trainer.metric_values['objective']
    assert objective[-1] < objective[0]
