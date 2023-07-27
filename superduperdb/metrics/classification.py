from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from superduperdb.core.metric import Metric
    from superduperdb.models.torch.wrapper import TorchModel


def compute_classification_metrics(
    validation_data: t.List[t.Dict[str, t.Any]],
    model: TorchModel,
    metrics: t.List[Metric],
) -> t.Dict[str, float]:
    X, y = model.training_keys
    out = {}
    predictions = model.predict([r[X] for r in validation_data])
    targets = t.cast(t.List[int], [r[y] for r in validation_data])
    assert all(isinstance(t, int) for t in targets)
    for m in metrics:
        out[m.identifier] = sum(
            [m(pred, target) for pred, target in zip(predictions, targets)]
        ) / len(validation_data)
    return out
