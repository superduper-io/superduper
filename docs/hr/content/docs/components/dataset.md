# `Dataset`

***Scope***

- An immutable snapshot of a query saved to `db.artifact_store`
- Used (currently) for validating model performance
- Snapshot allows users to know that their validation comparisons are apples-to-apples

***Usage pattern***

(Learn how to build a model [here](model))

```python
from superduperdb import Listener

ds = Dataset(
    'my-valid-data',
    select=query_selecting_data,
)

db.apply(ds)
```