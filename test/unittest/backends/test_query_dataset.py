import pytest

from superduperdb.backends.mongodb.query import Collection
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.components.model import Mapping

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_query_dataset(db):
    train_data = QueryDataset(
        db=db,
        mapping=Mapping('_base', signature='singleton'),
        select=Collection('documents').find(
            {}, {'_id': 0, 'x': 1, '_fold': 1, '_outputs': 1}
        ),
        fold='train',
    )
    r = train_data[0]

    assert '_id' not in r
    assert r['_fold'] == 'train'
    assert 'y' not in r

    assert r['_outputs']['x']['linear_a']['0'].shape[0] == 16

    train_data = QueryDataset(
        db=db,
        select=Collection('documents').find(),
        mapping=Mapping({'x': 'x', 'y': 'y'}, signature='**kwargs'),
        fold='train',
    )

    r = train_data[0]
    assert '_id' not in r
    assert set(r.keys()) == {'x', 'y'}


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_query_dataset_base(db):
    train_data = QueryDataset(
        db=db,
        select=Collection('documents').find({}, {'_id': 0}),
        mapping=Mapping(['_base', 'y'], signature='*args'),
        fold='train',
    )
    r = train_data[0]
    assert isinstance(r, tuple)
    assert len(r) == 2
