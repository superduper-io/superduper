import torch

from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.torch.model import TorchModel


def test_predict(database_with_random_tensor_data, database_with_float_tensors_32):
    encoder = database_with_random_tensor_data.encoders['torch.float32[32]']

    m = TorchModel(
        identifier='my-model',
        object=torch.nn.Linear(32, 7),
        encoder=encoder,
    )

    X = [
        r['x']
        for r in database_with_random_tensor_data.execute(
            Collection(name='documents').find()
        )
    ]

    out = m.predict(X=X, distributed=False)

    assert len(out) == len(X)

    m.predict(
        X='x',
        db=database_with_random_tensor_data,
        select=Collection(name='documents').find(),
        distributed=False,
    )
