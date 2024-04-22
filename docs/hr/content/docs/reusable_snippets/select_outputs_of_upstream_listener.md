# Select outputs of upstream listener

:::note
This is useful if you have performed a first step, such as pre-computing 
features, or chunking your data. You can use this query to 
operate on those outputs.
:::


```python
# <tab: MongoDB>
from superduperdb.backends.mongodb import Collection

indexing_key = upstream_listener.outputs_key
select = Collection(upstream_listener.outputs).find()
```


```python
# <tab: SQL>
indexing_key = upstream_listener.outputs_key
select = db.load("table", upstream_listener.outputs).to_query()
```
