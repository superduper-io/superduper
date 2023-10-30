import pytest

from superduperdb.backends.mongodb.query import Collection

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_find(random_data):
    r = random_data.execute(Collection('documents').find_one())
    print(r['x'].x.shape)

    cursor = random_data.execute(Collection('documents').find())
    print(next(cursor))
