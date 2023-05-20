from tests.fixtures.collection import with_semantic_index, random_data, float_tensors, empty, \
    a_model, a_watcher, an_update

import torch


def test_select(with_semantic_index):
    db = with_semantic_index.database
    r = next(db.execute_query('find', 'documents'))
    s = next(db.execute_query('find', 'documents', like={'x': r['x']},
                              semantic_index='test_learning_task', measure='css'))
    assert s['_id'] == r['_id']


def test_insert(random_data, a_watcher, an_update):
    random_data.database.execute_query(
        'insert_many',
        'documents',
        items=an_update,
    )
    r = random_data.find_one({'update': True})
    assert 'linear_a' in r['_outputs']['x']


def test_update(random_data, a_watcher):
    to_update = torch.randn(32)
    random_data.database.execute_query(
        'update_many',
        'documents',
        {},
        {'$set': {'x': to_update}}
    )

    r, s = list(random_data.find().limit(2))
    assert r['x'].tolist() == to_update.tolist()
    assert s['x'].tolist() == to_update.tolist()
    assert r['_outputs']['x']['linear_a'].tolist() == s['_outputs']['x']['linear_a'].tolist()


def test_delete(random_data):
    r = random_data.find_one()
    random_data.database.execute_query(
        'delete_many',
        'documents',
        {'_id': r['_id']}
    )
    assert random_data.find_one({'_id': r['_id']}) is None


def test_replace(random_data):
    r = random_data.find_one()
    x = torch.randn(32)
    r['x'] = x
    random_data.database.execute_query(
        'replace_one',
        'documents',
        {'_id': r['_id']},
        r,
    )
    r = random_data.find_one({'_id': r['_id']})
    assert r['x'].tolist() == x.tolist()


