# ruff: noqa: F401, F811
from superduperdb.datalayer.mongodb.query import Select
from tests.fixtures.collection import (
    empty,
    float_tensors,
    random_data,
    a_watcher,
    a_model,
    a_watcher_base,
    a_model_base,
)

from superduperdb.training.query_dataset import QueryDataset


def test_query_dataset(random_data, a_watcher):
    train_data = QueryDataset(
        select=Select(
            collection='documents',
            filter={},
            projection={'_id': 0, 'x': 1, '_fold': 1, '_outputs': 1},
        ),
        database='test_db',
        database_type='mongodb',
        fold='train',
        features={'x': 'linear_a'},
    )
    r = train_data[0]
    assert '_id' not in r
    assert r['_fold'] == 'train'
    assert 'y' not in r
    assert r['x'].shape[0] == 16

    train_data = QueryDataset(
        select=Select(
            collection='documents',
            filter={},
        ),
        database='test_db',
        database_type='mongodb',
        keys=['x', 'y'],
        fold='train',
    )

    r = train_data[0]
    assert '_id' not in r
    assert set(r.keys()) == {'x', 'y'}

    _ = QueryDataset(
        select=Select(
            collection='documents',
            filter={},
        ),
        database='test_db',
        database_type='mongodb',
        fold='valid',
    )


def test_query_dataset_base(random_data, a_watcher_base):
    train_data = QueryDataset(
        select=Select(
            collection='documents',
            filter={},
        ),
        database='test_db',
        database_type='mongodb',
        keys=['_base', 'y'],
        fold='train',
        features={'_base': 'linear_a_base'},
    )

    r = train_data[0]
    print(r)
