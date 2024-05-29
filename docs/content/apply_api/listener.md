# `Listener`

- apply a `model` to compute outputs on a query
- outputs are refreshed every-time new data are added
- outputs are saved to the `db.databackend`

***dependencies***

- [`Model`](./model.md)

***usage pattern***

(learn how to build a model [here](model))

```python
from superduperdb import Listener
m = ...  # build a model
q = ... # build a select query

# either...
listener = Listener(
    mode=m,
    select=q,
    key='x',
)

# or...
listener = m.to_listener(select=q, key='x')

db.apply(listener)
```

:::info
*how do i choose the `key` parameter?*
`key` refers to the field, or fields which 
will be fed into the `model` as `*args` and `**kwargs`

the following forms are possible:
- `key='x'`, 
- `key=('x','y')`, 
- `key={'x': 'x', 'y': 'y'}`, 
- `key=(('x',), {'y': 'y'})`,
:::

***see also***

- [change-data capture](../cluster_mode/change_data_capture)