# ruff: noqa: F401, F811
import torch

from superduperdb.core import Metric
from superduperdb.datalayer.mongodb.query import Select
from superduperdb.metrics.classification import compute_classification_metrics
from superduperdb.models.torch.wrapper import (
    TorchPipeline,
    TorchModel,
    TorchModelEnsemble,
)
from superduperdb.models.torch.wrapper import TorchTrainerConfiguration
from superduperdb.training.validation import validate_vector_search
from superduperdb.types.torch.tensor import tensor
from superduperdb.vector_search import VanillaHashSet

from tests.material.measures import css
from tests.fixtures.collection import (
    si_validation,
    empty,
    random_data,
    float_tensors_32,
    float_tensors_16,
    random_data_factory,
    metric,
)


class ToDict:
    def __init__(self):
        self.dict = dict(zip(list('abcdefghiklmnopqrstuvwyz'), range(26)))

    def __call__(self, input: str):
        return [self.dict[k] for k in input]


class TensorLookup:
    def __init__(self):
        self.d = torch.randn(26, 32)

    def __call__(self, x):
        return torch.stack([self.d[y] for y in x])


def pad_to_ten(x):
    to_stack = []
    for i, y in enumerate(x):
        out = torch.zeros(10, 32)
        y = y[:10]
        out[: y.shape[0], :] = y
        to_stack.append(out)
    return torch.stack(to_stack)


def test_pipeline():
    pl = TorchPipeline(
        'my-pipeline',
        [
            ('encode', ToDict()),
            ('lookup', TensorLookup()),
            ('forward', torch.nn.Linear(32, 256)),
            ('top1', lambda x: x.topk(1)[1]),
        ],
        collate_fn=pad_to_ten,
    )

    out = pl.predict('bla')

    print(out)

    assert isinstance(out, torch.Tensor)

    out = pl.predict(['bla', 'testing'], batch_size=2)

    assert isinstance(out, list)


def my_loss(X, y):
    return torch.nn.functional.binary_cross_entropy_with_logits(
        X[:, 0], y.type(torch.float)
    )


def acc(x, y):
    return x == y


def test_fit(random_data, si_validation):
    m = TorchModel(
        torch.nn.Linear(32, 1),
        'test',
        training_configuration=TorchTrainerConfiguration(
            optimizer_cls=torch.optim.Adam,
            identifier='my_configuration',
            objective=my_loss,
            loader_kwargs={'batch_size': 10},
            max_iterations=100,
            validation_interval=10,
            compute_metrics=compute_classification_metrics,
        ),
        postprocess=lambda x: int(torch.sigmoid(x).item() > 0.5),
    )
    m.fit(
        'x',
        'y',
        database=random_data,
        select=Select(collection='documents'),
        metrics=[Metric(identifier='acc', object=acc)],
        validation_sets=['my_valid'],
        serializer='dill',
    )


def ranking_loss(X):
    x, y = X
    x = x.div(x.norm(dim=1)[:, None])
    y = y.div(y.norm(dim=1)[:, None])
    similarities = x.matmul(y.T)  # causes a segmentation fault for no reason in pytest
    return -torch.nn.functional.log_softmax(similarities, dim=1).diag().mean()


def test_ensemble(si_validation, metric):
    encoder = tensor(torch.float, shape=(16,))
    a_model = TorchModel(torch.nn.Linear(32, 16), 'linear_a', encoder=encoder)
    c_model = TorchModel(torch.nn.Linear(32, 16), 'linear_c', encoder=encoder)

    config = TorchTrainerConfiguration(
        'ranking_task_parametrization',
        objective=ranking_loss,
        n_iterations=4,
        validation_interval=5,
        loader_kwargs={'batch_size': 10, 'num_workers': 0},
        optimizer_classes={
            'linear_a': torch.optim.Adam,
            'linear_c': torch.optim.Adam,
        },
        optimizer_kwargs={'lr': 0.001},
        compute_metrics=validate_vector_search,
        hash_set_cls=VanillaHashSet,
        measure=css,
        max_iterations=20,
    )

    m = TorchModelEnsemble(
        [a_model, c_model],
        identifier='my_ranking_ensemble',
    )

    m.fit(
        ['x', 'z'],
        training_configuration=config,
        database=si_validation,
        select=Select(collection='documents'),
    )
