import typing as t

from superduperdb.core.metric import Metric
from superduperdb.models.torch.wrapper import TorchModel


def compute_classification_metrics(
    validation_data: t.List, model: TorchModel, metrics: t.List[Metric]
) -> t.Dict[str, float]:
    X, y = model.training_keys
    out = {}
    predictions = model.predict([r[X] for r in validation_data])
    targets = [r[y] for r in validation_data]
    for m in metrics:
        out[m.identifier] = sum(
            [m(pred, target) for pred, target in zip(predictions, targets)]
        ) / len(validation_data)
    return out
