---
sidebar_position: 23
---

# Training models directly on your datastore

Similarly to [applying models to create predictions](./apply_models.md), training models is possible both procedurally and declaratively in `superduperdb`.

When models are trained, if `CFG.cluster.compute` has been configured (e.g. `"ray://ray-head:10001"`), then `superduperdb` deploys [a job on the configured `Ray` cluster](../production/non_blocking_ray_jobs.md)

## Basic pattern

### Procedural API

```python
model.fit(
    X='<input-col>',
    y='<target-col>',      # Optional, depending on whether supervised/ unsupervised,
    select=<query>,       # query which loads the training data
    db=db,
)
```

### Declarative API

```python
db.add(
    Model(
        *args, 
        training_select=<query>,   # to be passed as `Model.fit(..., select=...)`
        train_X='<input-col>',   # to be passed as `Model.fit(X=...)`
        train_y='<target-col>',   # to be passed as `Model.fit(..., y=...)`
        fit_kwargs={**...},   # kwargs to be passed to `Model.fit`
        **kwargs,
    )
)
```

## Fitting/ training models by framework

### Scikit-learn

```python
from superduperdb.ext.sklearn import Estimator
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
from superduperdb.ext.transformers import Pipeline
from superduperdb import superduper

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
from superduperdb.ext.torch import Module

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
    batch_size=100,  # any **kwargs supported by `superduperdb.ext.torch.TorchTrainerConfiguration`
    num_workers=4,
)
```