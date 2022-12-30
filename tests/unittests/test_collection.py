import torch

from tests.fixtures.collection import random_vectors, empty
from tests.material.converters import FloatTensor


def test_find_some(random_vectors):
    r = random_vectors.find_one()

    # check that the model has been used on new data to get outputs
    assert 'linear' in r['_outputs']['x']

    # check that the model outputs are of the expected type
    assert isinstance(r['_outputs']['x']['linear'], torch.Tensor)


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


def test_create_get_delete_x(empty):

    types_ = ['converter', 'loss', 'measure', 'metric', 'splitter']

    for type_ in types_:
        method = getattr(empty, f'create_{type_}')
        method(f'my_{type_}', func_)

    for type_ in types_:
        if type_ == 'loss':
            m = getattr(empty, 'losses')['my_loss']
        else:
            m = getattr(empty, f'{type_}s')[f'my_{type_}']

        # test that the retrieved object is the same
        assert m == func_
