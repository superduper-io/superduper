from test.torch import TorchModel, skip_torch, torch

from superduperdb.db.mongodb.query import Collection


@skip_torch
def test_predict(random_data, float_tensors_32):
    encoder = random_data.encoders['torch.float32[32]']

    m = TorchModel(
        identifier='my-model',
        object=torch.nn.Linear(32, 7),
        encoder=encoder,
    )

    X = [r['x'] for r in random_data.execute(Collection(name='documents').find())]

    out = m.predict(X=X, distributed=False)

    assert len(out) == len(X)

    m.predict(
        X='x',
        db=random_data,
        select=Collection(name='documents').find(),
        distributed=False,
    )
