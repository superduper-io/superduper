from test.db_config import DBConfig

import pytest
import torch.nn

from superduperdb.components.serializer import build_torch_state_serializer


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_custom_serializer(db):
    s = build_torch_state_serializer(
        torch.nn.Linear,
        {'in_features': 10, 'out_features': 32, 'bias': False},
    )

    m = torch.nn.Linear(10, 32, bias=False)
    bytes_ = s.encode(m)

    n = s.decode(bytes_)

    assert isinstance(n, torch.nn.Linear)
    assert n.weight.shape == m.weight.shape
    assert (n.weight.flatten() == m.weight.flatten()).sum().item() == m.weight.shape[
        0
    ] * m.weight.shape[1]

    db.add(s)

    s_reload = db.load('serializer', s.identifier)

    double_encoded = s_reload.decode(s_reload.encode(m))

    assert isinstance(double_encoded, torch.nn.Linear)
