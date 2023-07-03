# ruff: noqa: F401, F811
from superduperdb.datalayer.query_dataset import QueryDataset
from superduperdb.queries.mongodb.queries import Collection
from tests.fixtures.collection import (
    empty,
    random_data,
    float_tensors_16,
    float_tensors_32,
    random_data_factory,
    a_watcher,
    a_watcher_base,
    a_model,
    a_model_base,
)


def test_query_dataset(random_data, a_watcher):
    train_data = QueryDataset(
        select=Collection(name='documents').find(
            {}, {'_id': 0, 'x': 1, '_fold': 1, '_outputs': 1}
        ),
        fold='train',
        features={'x': 'linear_a'},
    )
    r = train_data[0]
    assert '_id' not in r
    assert r['_fold'] == 'train'
    assert 'y' not in r
    assert r['x'].shape[0] == 16

    train_data = QueryDataset(
        select=Collection(name='documents').find(),
        keys=['x', 'y'],
        fold='train',
    )

    r = train_data[0]
    assert '_id' not in r
    assert set(r.keys()) == {'x', 'y'}

    _ = QueryDataset(
        select=Collection(name='documents').find(),
        fold='valid',
    )


def test_query_dataset_base(random_data, a_watcher_base):
    train_data = QueryDataset(
        select=Collection(name='documents').find(),
        keys=['_base', 'y'],
        fold='train',
    )
    r = train_data[0]
    print(r)
