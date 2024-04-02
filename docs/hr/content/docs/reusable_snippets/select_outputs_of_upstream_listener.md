# Select outputs of upstream listener


```python
# <tab: MongoDB>
from superduperdb.backends.mongodb import Collection

select = Collection(upstream_listener.outputs).find()
```


```python
# <tab: SQL>
select = db.load('table', upstream_listener.outputs)
```
