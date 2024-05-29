# `Template`

- A `Component` containing placeholders flagged with `Variable` instances
- A `Template` may be used as the basis for applying multiple `Component` instances
- `Template` is leveraged by `Application`.
- Snapshot allows users to know that their validation comparisons are apples-to-apples
- A `Template` is useful for sharing, migrating and distributing AI components
- A `Template` may be applied to any SuperDuperDB deployment

***Usage pattern***

```python
from superduperdb import *

m = Listener(
    model=ObjectModel(
        object=lambda x: x + 2,
        identifier=Variable('model_id'),
    ),
    select=db['=collection'].find(),
    key=Variable('key')
)

# optional "info" parameter provides details about usage (depends on developer use-case)
t = Template(m, info={'key': {'type': 'str'}, 'collection': {'type': str}, 'model_id': {'type': 'str'}})

# doesn't trigger work/ computations
# just "saves" the template and its artifacts
db.apply(t) 

listener = t(key='my_key', collection='my_collection', model_id='my_id')

# this now triggers standard functionality
db.apply(listener)
```

***See also***

- [Application](./application.md)
