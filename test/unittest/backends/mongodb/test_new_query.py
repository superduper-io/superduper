from test.db_config import DBConfig

import pytest

from superduperdb import Document


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_add_fold(db):
    new_q = db['documents'].like(Document({'text': 'some like'})).limit(n=5)
    print(new_q)

    from superduperdb.ext.torch.encoder import tensor

    float_tensor = tensor(dtype='float', shape=(32,))
    import random

    import torch

    data = []
    for i in range(5):
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

    db.execute(
        db['documents'].insert_many(data),
        refresh=False,
    )

    x = db.execute(db['documents'].find_one()).unpack()

    assert x['x'].shape == (32,)
    assert x['y'] in [0, 1]
    assert x['z'].shape == (32,)
