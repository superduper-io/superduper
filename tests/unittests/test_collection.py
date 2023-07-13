import torch

from tests.fixtures.collection import random_vectors, empty, with_urls
from tests.material.converters import FloatTensor
from tests.material.losses import ranking_loss
from tests.material.metrics import accuracy, PatK
from tests.material.models import BinaryClassifier, BinaryTarget
from tests.material.splitters import trivial_splitter


def test_filter_content(random_vectors):
    filter_ = \
        {'item': {'_content': {'url': 'https://www.superduperdb.com/logos/white.png',
                               'converter': 'raw_bytes'}}}
    filter_ = random_vectors._get_content_for_filter(filter_)

    # check that "_content" in the filter has been downloaded
    assert isinstance(filter_['item'], bytes)


def test_find_some(random_vectors):

    r = random_vectors.find_one()
    print(r)

    # check that the model has been used on new data to get outputs
    assert 'linear' in r['_outputs']['x']
    assert 'other_linear' in r['_outputs']['x']
    assert 'model_attributes' in r['_outputs']['x']
    assert 'linear1' in r['_outputs']['x']['model_attributes']  # consider changing to `model_attributes/linear1`
    assert 'linear2' in r['_outputs']['x']['model_attributes']


def test_find_like(random_vectors):
    r_anchor = random_vectors.find_one()
    # retrieve same document with content
    r_found = random_vectors.find_one({'$like': {'document': {'x': r_anchor['x']}, 'n': 10}})
    assert r_anchor['_id'] == r_found['_id']

    # retrieve same document with exact match
    r_found = random_vectors.find_one({'$like': {'document': {'x': r_anchor['x']}, 'n': 10},
                                       'label': r_anchor['label']})
    assert r_anchor['_id'] == r_found['_id']

    # retrieve same document also finding similar first
    r_found = random_vectors.find_one({'$like': {'document': {'x': r_anchor['x']}, 'n': 10},
                                       'label': r_anchor['label']},
                                      similar_first=True)
    assert r_anchor['_id'] == r_found['_id']

    # retrieve same document with id
    r_found = random_vectors.find_one({'$like': {'document': {'_id': r_anchor['_id']}, 'n': 10}})
    assert r_anchor['_id'] == r_found['_id']

    random_vectors.semantic_index = 'other_linear'
    # test another semantic index with feature dependencies
    _ = random_vectors.find_one({'$like': {'document': {'x': r_anchor['x']}, 'n': 10}})

    random_vectors.remote = True
    # test that sending request to remote feature store works
    _ = random_vectors.find_one({'$like': {'document': {'x': r_anchor['x']}, 'n': 10}})


def test_replace(random_vectors):
    r = random_vectors.find_one(raw=True)
    r['a'] = 2
    random_vectors.replace_one({'_id': r['_id']}, r)


def test_insert(random_vectors):
    eps = 0.01
    x = torch.randn(32)
    y = torch.randn(32) * eps + x
    random_vectors.insert_one({
            'update': True,
            'i': 2000,
            'x': {
                '_content': {
                    'bytes': FloatTensor.encode(x),
                    'converter': 'float_tensor',
                }
            },
            'y': {
                '_content': {
                    'bytes': FloatTensor.encode(y),
                    'converter': 'float_tensor',
                }
            },
        })

    r = random_vectors.find_one({'update': True})

    # check that the model has been applied to the new datapoints
    assert 'linear' in r['_outputs']['x']

    random_vectors.remote = True
    job_ids = random_vectors.insert_one({
        'update': True,
        'i': 2001,
        'x': {
            '_content': {
                'bytes': FloatTensor.encode(x),
                'converter': 'float_tensor',
            }
        },
        'y': {
            '_content': {
                'bytes': FloatTensor.encode(y),
                'converter': 'float_tensor',
            }
        },
    })[1]

    # this blocks the program so that cleanup doesn't take place
    for model in job_ids:
        print(f'PROCESSING {model}...')
        for id_ in job_ids[model]:
            random_vectors.watch_job(id_)

    r = random_vectors.find_one({'i': 2001})

    # check that all models have been applied to the document
    assert len(r['_outputs']['x']) == 4


def test_update(random_vectors):

    r0 = random_vectors.find_one({'i': 0}, features={'x': 'linear'})
    r1 = random_vectors.find_one({'i': 1}, features={'x': 'linear'})

    random_vectors.update_one(
        {'i': 0},
        {'$set': {'x': {
            '_content': {
                'bytes': FloatTensor.encode(torch.randn(32)),
                'converter': 'float_tensor',
            }
        }}}
    )

    # check that the targeted documents were updated
    assert random_vectors.find_one({'i': 0},
                                   features={'x': 'linear'})['x'][0] != r0['x'][0]

    # check that other documents were not affected
    assert random_vectors.find_one({'i': 1},
                                   features={'x': 'linear'})['x'][0] == r1['x'][0]

def func_(x):
    return x + 2


def test_create_list_get_delete_x(empty):

    types_ = ['converter', 'loss', 'measure', 'metric', 'splitter']

    for type_ in types_:
        method = getattr(empty, f'create_{type_}')
        method(f'my_{type_}', func_)

    for type_ in types_:
        if type_ == 'loss':
            available = getattr(empty, 'list_losses')()
        else:
            available = getattr(empty, f'list_{type_}s')()
        assert available == [f'my_{type_}']

    for type_ in types_:
        if type_ == 'loss':
            m = getattr(empty, 'losses')['my_loss']
        else:
            m = getattr(empty, f'{type_}s')[f'my_{type_}']

        # test that the retrieved object is the same
        assert m == func_

    for type_ in types_:
        method = getattr(empty, f'delete_{type_}')
        method(f'my_{type_}', force=True)

    for type_ in types_:
        if type_ == 'loss':
            available = getattr(empty, 'list_losses')()
        else:
            available = getattr(empty, f'list_{type_}s')()

        # check that the objects have been deleted
        assert available == []


def test_create_delete_semantic_index(random_vectors):

    def f():
        return random_vectors.create_semantic_index(
            'ranking',
            models=[
                {
                    'name': 'encoder',
                    'object': torch.nn.Linear(32, 32),
                    'key': 'x',
                    'converter': 'float_tensor',
                    'active': True,
                },
                {
                    'name': 'identity',
                    'object': torch.nn.Identity(),
                    'key': 'y',
                    'converter': 'float_tensor',
                    'active': False,
                },
            ],
            loss={
                'name': 'ranking_loss',
                'object': ranking_loss,
            },
            filter={},
            metrics=[
                {
                    'name': 'p_at_1',
                    'object': PatK(1),
                },
            ],
            measure='css',
            batch_size=100,
            num_workers=0,
            n_epochs=100,
            lr=0.01,
            log_weights=True,
            download=True,
            validation_interval=5,
            n_iterations=10,
        )

    f()

    # check that the index has been registered
    assert 'ranking' in random_vectors.list_semantic_indexes()

    random_vectors.delete_semantic_index('ranking', force=True)
    random_vectors.delete_loss('ranking_loss', force=True)
    random_vectors.delete_metric('p_at_1', force=True)

    # check that the deletion was effective
    assert 'ranking' not in random_vectors.list_semantic_indexes()

    random_vectors.remote = True

    job_ids = f()
    print(job_ids)

    for job_id in job_ids:
        random_vectors.watch_job(job_id)


def test_create_self_supervised_index(random_vectors):

    random_vectors.create_semantic_index(
        'ranking',
        models=[
            {
                'name': 'encoder_self',
                'object': torch.nn.Linear(16, 16),
                'key': 'x',
                'converter': 'float_tensor',
                'active': True,
                'features': {'x': 'linear'},
            },
        ],
        loss={
            'name': 'ranking_loss',
            'object': ranking_loss,
        },
        filter={},
        metrics=[
            {
                'name': 'p_at_1',
                'object': PatK(1),
            },
        ],
        measure='css',
        splitter={'name': 'trivial', 'object': trivial_splitter},
        batch_size=100,
        num_workers=0,
        n_epochs=100,
        lr=0.01,
        log_weights=True,
        download=True,
        validation_interval=10,
        n_iterations=2,
    )


def test_create_delete_imputation(random_vectors):

    def f():
        return random_vectors.create_imputation(
            'classifier',
            model={
                'name': 'classifier',
                'object': BinaryClassifier(32),
                'key': 'x',
            },
            loss={
                'name': 'binary_loss',
                'object': torch.nn.BCEWithLogitsLoss(),
            },
            target={
                'name': 'label',
                'object': BinaryTarget(),
                'key': 'label'
            },
            metrics=[{
                'name': 'accuracy',
                'object': accuracy
            }],
            batch_size=20,
            num_workers=0,
            n_epochs=10,
            lr=0.001,
            validation_interval=5,
            n_iterations=10,
        )

    f()

    # check that the index has been registered
    assert 'classifier' in random_vectors.list_imputations()

    random_vectors.delete_imputation('classifier', force=True)
    random_vectors.delete_loss('binary_loss', force=True)
    random_vectors.delete_metric('accuracy', force=True)

    # check that deleting works
    assert 'classifier' not in random_vectors.list_imputations()

    random_vectors.remote = True

    job_ids = f()

    # test that the listing of jobs works
    assert set(random_vectors.list_jobs()) == set(job_ids)

    random_vectors.watch_job(job_ids[0])
    random_vectors.watch_job(job_ids[1])

    # check that the model has been registered
    assert 'classifier' in random_vectors.list_imputations()


def test_create_delete_neighbourhood(random_vectors):
    random_vectors.create_neighbourhood('test_sim', n=10)

    r = random_vectors.find_one()
    print(r)

    # test that similarities are in the record
    assert 'test_sim' in r.get('_like', {})
    assert len(r['_like']['test_sim']) == 10

    assert 'test_sim' in random_vectors.list_neighbourhoods()

    r = random_vectors.find_one(similar_join='test_sim')
    # test that joining the documents from the neighbourhood works
    assert isinstance(r['_like']['test_sim'][0], dict)

    random_vectors.delete_neighbourhood('test_sim', force=True)

    assert 'test_sim' not in random_vectors.list_neighbourhoods()

    r = random_vectors.find_one()
    print(r)
    assert 'test_sim' not in r.get('_like', {})

    random_vectors.remote = True

    job_id = random_vectors.create_neighbourhood('test_sim', n=7)

    random_vectors.watch_job(job_id)

    r = random_vectors.find_one()
    assert len(r['_like']['test_sim']) == 7


def test_downloads(with_urls):
    r = with_urls.find_one(raw=True)

    # check that the files/ content has been downloaded
    assert 'bytes' in r['item']['_content']


def test__convert_types(random_vectors):
    r = random_vectors._convert_types({
        'my_tensor': torch.randn(32)
    })

    # check that tensor has been converted to bytes
    assert isinstance(r['my_tensor']['_content']['bytes'], bytes)

