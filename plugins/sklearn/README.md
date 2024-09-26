<!-- Auto-generated content start -->
# superduper_sklearn

superduper allows users to work with arbitrary sklearn estimators, with additional support for pre-, post-processing and input/ output data-types.

## Installation

```bash
pip install superduper_sklearn
```

## API


- [Code](https://github.com/superduper-io/superduper/tree/main/plugins/sklearn)
- [API-docs](/docs/api/plugins/superduper_sklearn)

| Class | Description |
|---|---|
| `superduper_sklearn.model.SklearnTrainer` | A trainer for `sklearn` models. |
| `superduper_sklearn.model.Estimator` | Estimator model. |


## Examples

### Estimator

```python
from superduper_sklearn import Estimator
from sklearn.svm import SVC
model = Estimator(
    identifier='test',
    object=SVC(),
)
```


<!-- Auto-generated content end -->

<!-- Add your additional content below -->

## Training Example

Read more about this [here](https://docs.superduper.io/docs/templates/transfer_learning)
