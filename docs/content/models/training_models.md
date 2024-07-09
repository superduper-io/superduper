# Training models directly on your datastore

`Model` instances may be trained if a `trainer` is set on the `Model` when `db.apply` is called.
When models are trained, if `CFG.cluster.compute` has been configured with a `ray` scheduler, then `superduper` deploys [a job on the connected `ray` cluster](../production_features/non_blocking_ray_jobs).

## Basic pattern

```python
from superduper.ext.<framework> import <Framework>Trainer
from superduper.ext.<framework> import <ModelCls>

db.apply(
    <ModelCls>(
        *args, 
        trainer=<Framework>Trainer(**trainer_kwargs),
        **kwargs,
    )
)
```

## Fitting/ training models by framework

Not all `Model` types are trainable. We support training for the following frameworks:

| Framework | Training Link |
| --- | --- |
| Scikit-Learn | [link](../ai_integrations/sklearn#training) |
| PyTorch | [link](../ai_integrations/pytorch#training) |
| Transformers | [link](../ai_integrations/transformers#training) |

<!-- ### Scikit-learn

See [here]

```python
from superduper.ext.sklearn import Estimator
from sklearn.svm import SVC

m = Estimator(SVC(C=0.05))

m.fit(
    X='<input-col>',
    y='<target-col>',
    select=<query>,  # MongoDB, Ibis or SQL query
    db=db,
)
```

### Transformers

```python
from superduper.ext.transformers import Pipeline
from superduper import superduper

m = Pipeline(task='sentiment-analysis')

m.fit(
    X='<input-col>',
    y='<target-col>',
    db=db,
    select=<query>,   # MongoDB, Ibis or SQL query
    dataloader_num_workers=4,   # **kwargs are passed to `transformers.TrainingArguments`
)
```

### PyTorch

```python
import torch
from superduper.ext.torch import Module

model = Module(
    'my-classifier',
    preprocess=lambda x: torch.tensor(x),
    object=torch.nn.Linear(64, 512),
    postprocess=lambda x: x.topk(1)[0].item(),
)

model.fit(
    X='<input>',
    db=db,
    select=<query>,  # MongoDB, Ibis or SQL query
    batch_size=100,  # any **kwargs supported by `superduper.ext.torch.TorchTrainerConfiguration`
    num_workers=4,
)
``` -->