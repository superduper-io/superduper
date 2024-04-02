# Basic insertion

SuperDuperDB supports inserting data wrapped as dictionaries in Python.
These dictionaries may contain basic JSON-compatible data, but also 
other data-types to be handled with `DataType` components. All data inserts are wrapped with the `Document` wrapper:

```python
from superduperdb import Document
data = ... # an iterable of dictionaries
data = [Document(r) for r in data]
```

## MongoDB

```python
db.execute(
    my_collection.insert_many(data)
)
```

## SQL

```python
db.execute(
    my_table.insert(data)
)
```