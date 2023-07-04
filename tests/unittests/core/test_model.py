# ruff: noqa: F401, F811
import torch

from superduperdb.models.torch.wrapper import TorchModel
from superduperdb.datalayer.mongodb.query import Collection

from tests.fixtures.collection import (
    si_validation,
    empty,
    random_data,
    float_tensors_32,
    float_tensors_16,
    random_data_factory,
    metric,
)


def test_predict(random_data, float_tensors_32):
    encoder = random_data.types['torch.float32[32]']

    m = TorchModel(
        identifier='my-model',
        object=torch.nn.Linear(32, 7),
        encoder=encoder,
    )

    X = [r['x'] for r in random_data.execute(Collection(name='documents').find())]

    out = m.predict(X=X, remote=False)

    assert len(out) == len(X)

    m.predict(
        X='x',
        db=random_data,
        select=Collection(name='documents').find(),
        remote=False,
    )
