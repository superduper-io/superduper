# ruff: noqa: F401, F811
import torch
from torch.optim.adam import Adam

from superduperdb.core.metric import Metric
from superduperdb.metrics.classification import compute_classification_metrics
from superduperdb.models.torch.wrapper import (
    TorchPipeline,
    TorchModel,
    TorchModelEnsemble,
)
from superduperdb.models.torch.wrapper import TorchTrainerConfiguration
from superduperdb.metrics.vector_search import (
    VectorSearchPerformance,
    PatK,
)
from superduperdb.queries.mongodb.queries import Collection
from superduperdb.types.torch.tensor import tensor
from superduperdb.vector_search import VanillaVectorIndex

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

    out = pl._predict('bla')

    print(out)

    assert isinstance(out, torch.Tensor)

    out = pl._predict(['bla', 'testing'], batch_size=2)

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
            optimizer_cls=Adam,
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
        y='y',
        db=random_data,
        select=Collection(name='documents').find(),
        metrics=[Metric(identifier='acc', object=acc)],
        validation_sets=['my_valid'],
    )


def ranking_loss(x, y):
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
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={'lr': 0.001},
        compute_metrics=VectorSearchPerformance(
            measure='css',
            predict_kwargs={'batch_size': 10},
            index_key='x',
        ),
        hash_set_cls=VanillaVectorIndex,
        measure=css,
        max_iterations=20,
    )

    m = TorchModelEnsemble(
        [a_model, c_model],
        identifier='my_ranking_ensemble',
    )

    m.fit(
        X=['x', 'z'],
        configuration=config,
        db=si_validation,
        select=Collection(name='documents').find(),
        validation_sets=['my_valid'],
        metrics=[Metric('p@1', PatK(1))],
    )
