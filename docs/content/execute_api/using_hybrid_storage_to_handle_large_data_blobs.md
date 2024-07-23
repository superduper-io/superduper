---
sidebar_position: 16
---

# Working with and inserting large pieces of data

Some applications require large data-blobs and objects, which are either larger than the objects which are supported by the underlying database, or which will degrade performance of the database if stored directly. For example:

- large images
- large audio
- videos

In order to handle such data, superduper provides a few options when 
creating a `DataType` via the `encodable` parameter.

## Artifact store reference with `encodable='artifact'`

When creating a `DataType` with `encodable='artifact'`, 
the data encoded by the `DataType` is saved to the `db.artifact_store` 
and a reference in saved in the `db.databackend`

For example, if you try the following snippet:

```python
import pickle
import uuid
from superduper import DataType, Document, superduper, Table, Schema

db = superduper('mongomock://test', artifact_store='filesystem://./artifacts')

dt = DataType(
    'my-artifact',
    encoder=lambda x, info: pickle.dumps(x),
    decoder=lambda x, info: pickle.loads(x),
    encodable='artifact',
)

schema = Schema(identifier='schema', fields={'x': dt})
table = Table('my_collection', schema=schema)

db.apply(table)

my_id = str(uuid.uuid4())

db['my_collection'].insert_one(Document({'id': my_id, 'x': 'This is a test'})).execute()
```

If you now reload the data with this query:

```python
>>> r = db.execute(db['my_collection'].find_one({'id': my_id}))
>>> r
Document({'id': 'a9a01284-f391-4aaa-9391-318fc38303bb', 'x': 'This is a test', '_fold': 'train', '_id': ObjectId('669fae8ccdaeae826dec4784')})
```

You will see that `r['x']` is exactly `'This is a test'`, however, 
with a native MongoDB query, you will find the data for `'x'` missing:

```python
>>> db.databackend.conn.test.my_collection.find_one() 
{'id': 'a9a01284-f391-4aaa-9391-318fc38303bb',
 'x': '&:blob:866cf8526595d3620d6045172fb16d1efefac4b1',
 '_fold': 'train',
 '_schema': 'schema',
 '_builds': {},
 '_files': {},
 '_blobs': {},
 '_id': ObjectId('669fae8ccdaeae826dec4784')}
```

This is because the data is stored in the filesystem/ artifact store `./artifacts`.
You may verify that with this command:

```bash
iconv -f ISO-8859-1 -t UTF-8 artifacts/866cf8526595d3620d6045172fb16d1efefac4b1
```

The superduper query reloads the data and passes it to the query result, 
without any user intervention.

## Just-in-time loading with `encodable='lazy_artifact'`:

If you specify `encodable='lazy_artifact'`, then the data 
is only loaded when a user calls the `.unpack()` method.
This can be useful if the datapoints are very large, 
and should only be loaded when absolutely necessary.

Try replacing the creation of `dt` with this command:

```python
dt = DataType(
    'my-artifact',
    encoder=lambda x, info: pickle.dumps(x),
    decoder=lambda x, info: pickle.loads(x),
    encodable='lazy_artifact',
)
```

and then execute the same lines as before.
You will find that:

```python
>>> r = db.execute(my_collection.find_one({'id': my_id}))
>>> r
Document({'id': 'b2a248c7-e023-4cba-9ac9-fdc92fa77ae3', 'x': LazyArtifact(identifier='', uuid='c0db12ad-2684-4e39-a2ba-2748bd20b193', datatype=DataType(identifier='my-artifact', uuid='6d72b346-b5ec-4d8b-8cba-cddec86937a3', upstream=None, plugins=None, encoder=<function <lambda> at 0x125e33760>, decoder=<function <lambda> at 0x125c4e320>, info=None, shape=None, directory=None, encodable='lazy_artifact', bytes_encoding='Bytes', intermediate_type='bytes', media_type=None), uri=None, x=<EMPTY>), '_fold': 'train', '_id': ObjectId('669faf9dcdaeae826dec4789')})
>>> r['x'].x
<EMPTY>
```

However, after calling `.unpack(db)`:

```python
>>> r = r.unpack()
>>> r['x']
'This is a test'
```

This allows `superduper` to build efficient data-loaders and model loading mechanisms.
For example, when saving model data to the artifact-store, the default `encodable` is `'lazy_artifact'`.

## Saving files and directories to the artifact store

There is an additional mechanism for working with large files. This works 
better in certain contexts, such as flexibly saving the results of model training.
The following lines copy the file to the `db.artifact_store`.
When data is loaded, the data is copied back over from the artifact-store to 
the local file-system:

```bash
cp -r test test_copy
```

```python
schema = Schema(identifier='schema', fields={'x': dt})
table = Table('my_collection', schema=schema)

db.apply(table)
my_id = str(uuid.uuid4())
db.execute(db['my_collection'].insert_one(Document({'id': my_id, 'x': './test_copy'})))
```

When reloading data, you will see that only a reference to the data in the artifact-store
is loaded:

```python
>>> db.execute(db['my_collection'].find_one({'id': my_id})).unpack()
{'id': '93eaae04-a48b-4632-94cf-123cdb2c9517',
 'x': './artifacts/d537309c8e5be28f91b90b97bbb229984935ba4a/test_copy',
 '_fold': 'train',
 '_id': ObjectId('669fb091cdaeae826dec4797')}

```

Downstream `Model` instances may then explicitly handle the local file from the file 
reference.