from tests.fixtures.collection import empty, float_tensors, random_data, a_watcher, a_model, a_watcher_base, a_model_base

from superduperdb.training.loading import QueryDataset


def test_query_dataset(random_data, a_watcher):

    train_data = QueryDataset(
        database_type='mongodb',
        database='test_db',
        query_params=('documents', {}, {'_id': 0, 'x': 1, '_fold': 1, '_outputs': 1}),  # can also include projection as third parameter
        fold='train',
        features={'x': 'linear_a'}
    )

    r = train_data[0]
    assert '_id' not in r
    assert r['_fold'] == 'train'
    assert 'y' not in r
    assert r['x'].shape[0] == 16

    train_data = QueryDataset(
        database_type='mongodb',
        database='test_db',
        query_params=('documents', {}),  # can also include projection as third parameter
        fold='train',
        keys=['x', 'y']
    )

    r = train_data[0]
    assert '_id' not in r
    assert set(r.keys()) == {'x', 'y'}

    valid_data = QueryDataset(
        database_type='mongodb',
        database='test_db',
        query_params=('documents', {}),  # can also include projection as third parameter
        fold='valid',
    )

    print(len(valid_data))


def test_query_dataset_base(random_data, a_watcher_base):

    train_data = QueryDataset(
        database_type='mongodb',
        database='test_db',
        query_params=('documents', {}),
        keys=['_base', 'y'],
        fold='train',
        features={'_base': 'linear_a_base'}
    )

    r = train_data[0]
    print(r)
