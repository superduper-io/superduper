# Predictions

Model predictions may be deployed by calling `Model.predict` or `Model.predict_one` directly.

```python
m = db.load('model', 'my-model')

# *args, **kwargs depend on model implementation
results = m.predict_one(*args, **kwargs)
```

An alternative is to construct a prediction "query" as follows:

```python
from superduperdb.backends.base.query import models

# *args, **kwargs depend on model implementation
q = models['my-model'].predict_one(*args, **kwargs)
```

The query may then be "executed" on the datalayer `db`:

```python
results = db.execute(q)
```

The results should be the same for both versions.