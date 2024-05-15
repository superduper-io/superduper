import pytest

from superduperdb.backends.mongodb.query import MongoQuery
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
        select=MongoQuery(identifier='documents', db=db).find(
            {}, {'_id': 0, 'x': 1, '_fold': 1, '_outputs': 1, '_leaves': 1, '_blobs': 1, '_files': 1}
        ),
        fold='train',
    )
    r = train_data[0]
    assert '_id' not in r
    assert r['_fold'] == 'train'
    assert 'y' not in r

    key = [l.split('/')[-1] for l in db.show('listener') if db.load('listener', l).key =='x' ][0]
    assert r['_outputs'][key].shape[0] == 16

    train_data = QueryDataset(
        db=db,
        select=MongoQuery('documents').find(),
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
        select=MongoQuery('documents').find({}, {'_id': 0}),
        mapping=Mapping(['_base', 'y'], signature='*args'),
        fold='train',
    )
    r = train_data[0]
    assert isinstance(r, tuple)
    assert len(r) == 2
