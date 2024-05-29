# Basic insertion

SuperDuperDB supports inserting data wrapped as dictionaries in Python.
These dictionaries may contain basic JSON-compatible data, but also 
other data-types to be handled with `DataType` components. All data inserts are wrapped with the `Document` wrapper:

```python
data = ... # an iterable of dictionaries
```

For example, first get some [sample data](../reusable_snippets/get_useful_sample_data.md):

```bash
!curl -O https://superduperdb-public-demo.s3.amazonaws.com/text.json
```

Then load the data:

```python
with open('./text.json') as f:
    data = json.load(f)
```

## Usage pattern

```python
ids, jobs = db['collection-name'].insert(data).execute()
```

## MongoDB

```python
ids, jobs = db['collection-name'].insert_many(data).execute()
```

A `Schema` which differs from the standard `Schema` used by `"collection-name"` may 
be used with:

```python
ids, jobs = db['collection-name'].insert_many(data).execute(schema=schema_component)
```

Read about this here `Schema` [here](../apply_api/schema.md).

## SQL

```python
ids, jobs = db['table-name'].insert(data)
```
If no `Schema` has been set-up for `table-name"` a `Schema` is auto-inferred.
Data not handled by the `db.databackend` is encoded by default with `pickle`.

## Monitoring jobs

The second output of this command gives a reference to the job computations 
which are triggered by the `Component` instances already applied with `db.apply(...)`.

If users have configured a `ray` cluster, the jobs may be monitored at the 
following uri:

```python
from superduperdb import CFG

print(CFG.cluster.compute.uri)
```