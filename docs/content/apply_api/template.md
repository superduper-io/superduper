# `Template`

- A `Component` containing placeholders flagged with `Variable` instances
- A `Template` may be used as the basis for applying multiple `Component` instances
- `Template` is leveraged by `Application`.
- Snapshot allows users to know that their validation comparisons are apples-to-apples
- A `Template` is useful for sharing, migrating and distributing AI components
- A `Template` may be applied to any SuperDuperDB deployment

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

***See also***

- [Application](./application.md)
