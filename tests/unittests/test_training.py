import torch

from sddb.training.losses import NegativeLoss
from tests.material.converters import FloatTensor
from tests.material.metrics import accuracy

from tests.material.models import Dummy, DummyClassifier, DummyTarget
from tests.fixtures.collection import collection_no_hashes


def dot(x, y):
    return x.matmul(y.T)


def css(x, y):
    x = x.div(x.pow(2).sum(1).sqrt()[:, None])
    y = y.div(y.pow(2).sum(1).sqrt()[:, None])
    return dot(x, y)


def simple_split(x):
    output0 = x.copy()
    output1 = x.copy()
    output0['test'] = x['test'].split(' ')[0]
    output1['test'] = x['test'].split(' ')[1]
    return output0, output1


def p_at_1(x, y):
    return sum([y[i] in x[i] for i in range(len(y))]) / len(y)


def test_representation_trainer(collection_no_hashes):
    collection_no_hashes.create_semantic_index(
        'my_semantic_index',
        measure={'name': 'css', 'object': css},
        loss={'name': 'negative_loss', 'object': NegativeLoss()},
        models=[{'name': 'my_model', 'object': Dummy(),  'key': '_base', 'filter': {},
                 'active': True, 'converter': {'name': 'float_tensor', 'object': FloatTensor()}}],
        splitter={'name': 'simple', 'object': simple_split},
        metrics=[{'name': 'p@1', 'object': p_at_1}],
        n_epochs=2,
        batch_size=2,
        num_workers=0,
    )

    assert 'my_semantic_index' in collection_no_hashes.list_semantic_indexes()
    _ = collection_no_hashes.models['my_model']
    h = collection_no_hashes.hash_set
    assert h.shape[0] == 10


def test_imputation_trainer(collection_no_hashes):
    collection_no_hashes.create_imputation(
        'my_imputation',
        model=DummyClassifier(),
        target={'name': 'target', 'object': DummyTarget(), 'key': 'fruit'},
        loss={'name': 'cross_entropy', 'object': torch.nn.CrossEntropyLoss()},
        metrics=[{'name': 'accuracy', 'object': accuracy}],
        key='_base',
        batch_size=2,
        num_workers=0,
        n_epochs=5,
    )
    assert 'my_imputation' in collection_no_hashes.list_models()
    _ = collection_no_hashes.models['my_imputation']
    assert collection_no_hashes.count_documents(
        {'_outputs._base.my_imputation': {'$exists': 1}}
    ) == 10
