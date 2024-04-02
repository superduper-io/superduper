# Select outputs of upstream listener

:::note
This is useful if you have performed a first step, such as pre-computing 
features, or chunking your data. You can use this query to 
operate on those outputs.
:::


```python
# <tab: MongoDB>
from superduperdb.backends.mongodb import Collection

select = Collection(upstream_listener.outputs).find()
```


```python
# <tab: SQL>
select = db.load('table', upstream_listener.outputs)
```
