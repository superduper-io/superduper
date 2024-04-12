---
sidebar_position: 16
---

# Working with and inserting large pieces of data

Some applications require large data-blobs and objects, which are either larger than the objects which are supported by the underlying database, or which will degrade performance of the database if stored directly. For example:

- large images
- large audio
- videos

In order to handle such data, SuperDuperDB provides a few options when 
creating a `DataType` via the `encodable` parameter.

## Artifact store reference with `encodable='artifact'`

When creating a `DataType` with `encodable='artifact'`, 
the data encoded by the `DataType` is saved to the `db.artifact_store` 
and a reference in saved in the `db.databackend`

For example, if you try the following snippet:

```python
import pickle
import uuid
from superduperdb import DataType, Document, superduper
from superduperdb.backends.mongodb import Collection

db = superduper('mongomock://test', artifact_store='filesystem://./artifacts')

dt = DataType(
    'my-artifact',
    encoder=lambda x, info: pickle.dumps(x),
    decoder=lambda x, info: pickle.loads(x),
    encodable='artifact',
)

db.apply(dt)

my_collection = Collection('my_collection')

my_id = str(uuid.uuid4())

db.execute(my_collection.insert_one(Document({'id': my_id, 'x': dt('This is a test')})))
```

If you now reload the data with this query:

```python
>>> r = db.execute(my_collection.find_one({'id': my_id}))
>>> r
Document({'id': '9458a837-3192-43a0-8c27-e2fbe72de74c', 'x': Artifact(file_id='866cf8526595d3620d6045172fb16d1efefac4b1', datatype=DataType(identifier='my-artifact', encoder=<function <lambda> at 0x15739e700>, decoder=<function <lambda> at 0x15739e520>, info=None, shape=None, directory=None, encodable='artifact', bytes_encoding=<BytesEncoding.BYTES: 'Bytes'>, media_type=None), uri=None, sha1=None, x='This is a test', artifact=False), '_fold': 'train', '_id': ObjectId('661aecc8ecd56c75bfd3add3')})
>>> r.unpack()
{'id': '9458a837-3192-43a0-8c27-e2fbe72de74c',
 'x': 'This is a test',
 '_fold': 'train',
 '_id': ObjectId('661aecc8ecd56c75bfd3add3')}
```

You will see that `r['x']` is exactly `'This is a test'`, however, 
with a native MongoDB query, you will find the data for `'x'` missing:

```python
>>> db.databackend.conn.test.my_collection.find_one() 
{'id': '9458a837-3192-43a0-8c27-e2fbe72de74c',
 'x': {'_content': {'bytes': None,
   'datatype': 'my-artifact',
   'leaf_type': 'artifact',
   'sha1': None,
   'uri': None,
   'file_id': '866cf8526595d3620d6045172fb16d1efefac4b1'}},
 '_fold': 'train',
 '_id': ObjectId('661aecc8ecd56c75bfd3add3')}
```

This is because the data is stored in the filesystem/ artifact store `./artifacts`.
You may verify that with this command:

```bash
iconv -f ISO-8859-1 -t UTF-8 artifacts/866cf8526595d3620d6045172fb16d1efefac4b1
```

The SuperDuperDB query reloads the data and passes it to the query result, 
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
    encodable='lazy-artifact',
)
```

and then execute the same lines as before.
You will find that:

```python
>>> r = db.execute(my_collection.find_one({'id': my_id}))
>>> r['x'].x
<EMPTY>
```

However, after calling `.unpack(db)`:

```python
>>> r = r.unpack(db)
>>> r['x']
'This is a test'
```

This allows `superduperdb` to build efficient data-loaders and model loading mechanisms.
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
dt = DataType('my-file', encodable='file')
db.apply(dt)
my_id = str(uuid.uuid4())
db.execute(my_collection.insert_one(Document({'id': my_id, 'x': dt('./test_copy')})))
```

When reloading data, you will see that only a reference to the data in the artifact-store
is loaded:

```python
>>> db.execute(my_collection.find_one({'id': my_id})).unpack(db)
{'id': '2b14133a-f275-461e-b0a2-d6f0eadb8b9b',
 'x': './artifacts/4dc048d4dbf67bed983a1b7a82822347645cc240',
 '_fold': 'train',
 '_id': ObjectId('661b9c229a2e44f23aa16422')}
```

Downstream `Model` instances may then explicitly handle the local file from the file 
reference.