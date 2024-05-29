# `Metric`

- Wrapper around a function intended to validate model outputs
- Function returns scalar value
- Used in `Validation`, `Model` and `Trainer` to measure `Model` performance

***Usage pattern***

```python
from superduperdb import Metric

def example_comparison(x, y):
    return sum([xx == yy for xx, yy in zip(x, y)]) / len(x)

m = Metric(
    'accuracy',
    object=example_comparison,
)

db.apply(m)
```

***See also***

- [Change-data capture](../cluster_mode/change_data_capture)
- [Validation](./validation.md)