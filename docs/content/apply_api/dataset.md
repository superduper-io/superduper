# `Dataset`

- An immutable snapshot of a query saved to `db.artifact_store`
- Used (currently) for validating model performance
- Snapshot allows users to know that their validation comparisons are apples-to-apples

***Usage pattern***

(Learn how to build a model [here](model))

```python
from superduperdb import Listener

ds = Dataset(
    'my-valid-data',
    select=db['my_table'].select(),   # `.select()` selects whole table
)

db.apply(ds)
```

***Explanation***

- On creation `superduperdb` queries the data from the `db.databackend` based on the `select` parameter.
- The data queries like this is saved as a persistent blob in the `db.artifact_store`.
- When the dataset is reloaded, the `select` query is not executed again, instead the 
  data is reloaded from the `db.artifact_store`. This ensures the `Dataset` is always "the same".
- `Dataset` is handy for making sure model validations are comparable.