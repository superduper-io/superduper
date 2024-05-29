# `Validation`

- Validate a `Model` by attaching a `Validation` component

***Dependencies***

- [`Metric`](metric.md)
- [`Dataset`](./dataset.md)

***Usage pattern***

```python
from superduperdb import Validation

validation = Validation(
    datasets=[dataset_1, ...],
    metrics=[metric_1, ...],
    key=('X', 'y')    # key to use for the comparison
)

model = Model(
    ...     # standard arguments
    validation=validation,
)

# Applying model recognizes `.validation` attribute
# and validates model on the `.datasets` with `.metrics`
db.apply(model)
```