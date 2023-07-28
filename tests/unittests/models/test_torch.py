import torch

from superduperdb.core.metric import Metric
from superduperdb.datalayer.mongodb.query import Collection
from superduperdb.metrics.classification import compute_classification_metrics
from superduperdb.models.torch.wrapper import TorchModel, TorchTrainerConfiguration


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


def my_loss(X, y):
    return torch.nn.functional.binary_cross_entropy_with_logits(
        X[:, 0], y.type(torch.float)
    )


def acc(x, y):
    return x == y


def test_fit(random_data, si_validation):
    m = TorchModel(
        object=torch.nn.Linear(32, 1),
        identifier='test',
        training_configuration=TorchTrainerConfiguration(
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
        X='x',
        y='y',
        db=random_data,
        select=Collection(name='documents').find(),
        metrics=[Metric(identifier='acc', object=acc)],
        validation_sets=['my_valid'],
    )
