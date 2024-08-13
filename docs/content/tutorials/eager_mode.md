
# Eager Mode (Aplpa) 

Eager Mode is an interactive way to build Superduper applications. 

Users can input data as usual, continuously call models, and view results.
Once the interactive debugging and construction are complete, 
the corresponding data pipeline can be built directly through `apply`, 
eliminating the need for direct debugging between AI application models and databases.

:::note
This feature is in the alpha version.
:::

Connect the database and insert data.

```python
from superduper import superduper

db = superduper('mongomock://test')
```

<details>
<summary>Outputs</summary>
<pre>
    2024-Jul-25 11:20:56.95| INFO     | zhouhaha-2.local| superduper.base.build:57   | Data Client is ready. mongomock.MongoClient('localhost', 27017)
    2024-Jul-25 11:20:56.96| INFO     | zhouhaha-2.local| superduper.base.build:36   | Connecting to Metadata Client with engine:  mongomock.MongoClient('localhost', 27017)
    2024-Jul-25 11:20:56.96| INFO     | zhouhaha-2.local| superduper.base.build:152  | Connecting to compute client: Compute(uri=None, compute_kwargs=\{\}, _path='superduper.backends.local.compute.LocalComputeBackend')
    2024-Jul-25 11:20:56.96| INFO     | zhouhaha-2.local| superduper.base.datalayer:105  | Building Data Layer
    2024-Jul-25 11:20:56.96| INFO     | zhouhaha-2.local| superduper.base.build:214  | Configuration: 
     +---------------+------------------+
    | Configuration |      Value       |
    +---------------+------------------+
    |  Data Backend | mongomock://test |
    +---------------+------------------+

</pre>
</details>

```python
import numpy as np
data = [
    {"x": 1, "y": "2", "z": np.array([1, 2, 3])},
    {"x": 2, "y": "3", "z": np.array([4, 5, 6])},
    {"x": 3, "y": "4", "z": np.array([7, 8, 9])},
]

db["documents"].insert(data).execute()
```

<details>
<summary>Outputs</summary>
<pre>
    2024-Jul-25 11:20:57.00| INFO     | zhouhaha-2.local| superduper.base.datalayer:426  | Table documents does not exist, auto creating...
    2024-Jul-25 11:20:57.00| INFO     | zhouhaha-2.local| superduper.base.datalayer:432  | Creating table documents with schema \{('z', 'numpy-int64[3]')\}
    2024-Jul-25 11:20:57.00| WARNING  | zhouhaha-2.local| superduper.misc.annotations:119  | add is deprecated and will be removed in a future release.

</pre>
<pre>
    ([ObjectId('66a1c49909d51b458635eda2'),
      ObjectId('66a1c49909d51b458635eda3'),
      ObjectId('66a1c49909d51b458635eda4')],
     None)
</pre>
</details>

When using `select.execute(eager_mode=True)`, all returned data will enter eager mode, which can be used for interactive model pipeline construction.

```python
data = list(db["documents"].select().execute(eager_mode=True))[0]
data
```

<details>
<summary>Outputs</summary>
<pre>
    Document(\{'x': 1, 'y': '2', 'z': array([1, 2, 3]), '_fold': 'train', '_id': ObjectId('66a1c49909d51b458635eda2')\})
</pre>
</details>

Define the first model and make predictions.

```python
from superduper import ObjectModel
def func_a(x):
    return {"x": x, "model": "a"}

model_a = ObjectModel(identifier="a", object=func_a)
output_a = model_a(data["x"])
output_a
```

<details>
<summary>Outputs</summary>
<pre>
    \{'x': 1, 'model': 'a'\}
</pre>
</details>

Define the second model and make predictions.

```python
def func_b(x, y, o_a):
    return {
        "x": x,
        "y": y,
        "o_a": o_a,
        "model": "b"
    }

model_b = ObjectModel(identifier="b", object=func_b)
output_b = model_b(data["x"], data["y"], output_a)
output_b
```

<details>
<summary>Outputs</summary>
<pre>
    \{'x': 1, 'y': '2', 'o_a': \{'x': 1, 'model': 'a'\}, 'model': 'b'\}
</pre>
</details>

Define the third model and make predictions.

```python
def func_c(x, y, z, o_a, o_b):
    return {
        "x": x,
        "y": y,
        "z": z,
        "o_a": o_a,
        "o_b": o_b,
        "model": "c",
    }

model_c = ObjectModel(identifier="c", object=func_c)
output_c = model_c(data["x"], data["y"], data["z"], output_a, output_b)
output_c
```

<details>
<summary>Outputs</summary>
<pre>
    \{'x': 1, 'y': '2', 'z': array([1, 2, 3]), 'o_a': \{'x': 1, 'model': 'a'\}, 'o_b': \{'x': 1, 'y': '2', 'o_a': \{'x': 1, 'model': 'a'\}, 'model': 'b'\}, 'model': 'c'\}
</pre>
</details>

Apply all models to the data to start monitoring the data and making predictions.
When adding a model result, not only the current model but also the recursively dependent upstream models will be added.

```python
output_c.apply()
```

<details>
<summary>Outputs</summary>
<pre>
    2024-Jul-25 11:20:57.14| INFO     | zhouhaha-2.local| superduper.misc.eager:289  | Applying nodes: ['a', 'b', 'c']
    2024-Jul-25 11:20:57.14| INFO     | zhouhaha-2.local| superduper.misc.eager:301  | Applying node: a
    2024-Jul-25 11:20:57.14| INFO     | zhouhaha-2.local| superduper.misc.eager:302  | Example output \{'x': 1, 'model': 'a'\}
    2024-Jul-25 11:20:57.14| INFO     | zhouhaha-2.local| superduper.misc.eager:306  | Key: x
    2024-Jul-25 11:20:57.14| INFO     | zhouhaha-2.local| superduper.misc.eager:309  | Select: documents.find(\{\}, \{'x': 1, '_builds': 1, '_files': 1, '_blobs': 1, '_schema': 1\})
    2024-Jul-25 11:20:57.14| INFO     | zhouhaha-2.local| superduper.misc.eager:311  | Predict id: a
    2024-Jul-25 11:20:57.14| INFO     | zhouhaha-2.local| superduper.misc.eager:313  | Predict kwargs: \{\}
    2024-Jul-25 11:20:57.14| INFO     | zhouhaha-2.local| superduper.misc.eager:323  | Listener: Listener(identifier='a', uuid='a', upstream=None, plugins=None, key='x', model=ObjectModel(identifier='a', uuid='ea5db97f-ab09-4804-9566-14c57f84f705', upstream=None, plugins=None, signature='*args,**kwargs', datatype=None, output_schema=None, flatten=False, model_update_kwargs=\{\}, predict_kwargs=\{\}, compute_kwargs=\{\}, validation=None, metric_values=\{\}, num_workers=0, object=\<function func_a at 0x146fa7d90\>), select=documents.find(\{\}, \{'x': 1, '_builds': 1, '_files': 1, '_blobs': 1, '_schema': 1\}), predict_kwargs=\{\})
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.components.listener:83   | Requesting listener setup on CDC service
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.components.listener:93   | Skipping listener setup on CDC service since no URI is set
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.jobs.queue:104  | Declaring component listener/a
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.jobs.queue:178  | Running jobs for listener::a
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.backends.local.compute:64   | Submitting job. function:\<function method_job at 0x108f1fd90\>
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.components.model:678  | Requesting prediction in db - [a] with predict_id a
    
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.components.model:807  | Adding 3 model outputs to `db`
    2024-Jul-25 11:20:57.15| SUCCESS  | zhouhaha-2.local| superduper.backends.local.compute:70   | Job submitted on \<superduper.backends.local.compute.LocalComputeBackend object at 0x146f38ee0\>.  function:\<function method_job at 0x108f1fd90\> future:e6caa179-a9f1-42de-91d2-7842e11143b2
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.misc.eager:301  | Applying node: b
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.misc.eager:302  | Example output \{'x': 1, 'y': '2', 'o_a': \{'x': 1, 'model': 'a'\}, 'model': 'b'\}
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.misc.eager:306  | Key: ('x', 'y', '_outputs.a')
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.misc.eager:309  | Select: MongoOutputs(identifier='MongoOutputs-identifier-uuid-853b05d5-1206-47d3-b3f6-93fcd66be97f-table-documents-parts-find-x-1-y-1-builds-1-files-1-blobs-1-schema-1-outputs-a', uuid='853b05d5-1206-47d3-b3f6-93fcd66be97f', table='documents', parts=[('find', (\{\}, \{'x': 1, 'y': 1, '_builds': 1, '_files': 1, '_blobs': 1, '_schema': 1\}), \{\}), ('outputs', ('a',), \{\})])
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.misc.eager:311  | Predict id: b
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.misc.eager:313  | Predict kwargs: \{\}
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.misc.eager:323  | Listener: Listener(identifier='b', uuid='b', upstream=None, plugins=None, key=('x', 'y', '_outputs.a'), model=ObjectModel(identifier='b', uuid='5fe3670b-c49b-4999-b736-5af3c584f299', upstream=None, plugins=None, signature='*args,**kwargs', datatype=None, output_schema=None, flatten=False, model_update_kwargs=\{\}, predict_kwargs=\{\}, compute_kwargs=\{\}, validation=None, metric_values=\{\}, num_workers=0, object=\<function func_b at 0x146fa7e20\>), select=MongoOutputs(identifier='MongoOutputs-identifier-uuid-853b05d5-1206-47d3-b3f6-93fcd66be97f-table-documents-parts-find-x-1-y-1-builds-1-files-1-blobs-1-schema-1-outputs-a', uuid='853b05d5-1206-47d3-b3f6-93fcd66be97f', table='documents', parts=[('find', (\{\}, \{'x': 1, 'y': 1, '_builds': 1, '_files': 1, '_blobs': 1, '_schema': 1\}), \{\}), ('outputs', ('a',), \{\})]), predict_kwargs=\{\})
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.components.listener:83   | Requesting listener setup on CDC service
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.components.listener:93   | Skipping listener setup on CDC service since no URI is set
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.jobs.queue:104  | Declaring component listener/b
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.jobs.queue:178  | Running jobs for listener::b
    2024-Jul-25 11:20:57.15| INFO     | zhouhaha-2.local| superduper.backends.local.compute:64   | Submitting job. function:\<function method_job at 0x108f1fd90\>
    2024-Jul-25 11:20:57.16| INFO     | zhouhaha-2.local| superduper.components.model:678  | Requesting prediction in db - [b] with predict_id b
    
    2024-Jul-25 11:20:57.16| INFO     | zhouhaha-2.local| superduper.components.model:807  | Adding 3 model outputs to `db`
    2024-Jul-25 11:20:57.16| SUCCESS  | zhouhaha-2.local| superduper.backends.local.compute:70   | Job submitted on \<superduper.backends.local.compute.LocalComputeBackend object at 0x146f38ee0\>.  function:\<function method_job at 0x108f1fd90\> future:c612486d-58ea-4654-a012-5ca3cfc86abe
    2024-Jul-25 11:20:57.16| INFO     | zhouhaha-2.local| superduper.misc.eager:301  | Applying node: c
    2024-Jul-25 11:20:57.16| INFO     | zhouhaha-2.local| superduper.misc.eager:302  | Example output \{'x': 1, 'y': '2', 'z': array([1, 2, 3]), 'o_a': \{'x': 1, 'model': 'a'\}, 'o_b': \{'x': 1, 'y': '2', 'o_a': \{'x': 1, 'model': 'a'\}, 'model': 'b'\}, 'model': 'c'\}
    2024-Jul-25 11:20:57.16| INFO     | zhouhaha-2.local| superduper.misc.eager:306  | Key: ('x', 'y', 'z', '_outputs.a', '_outputs.b')
    2024-Jul-25 11:20:57.16| INFO     | zhouhaha-2.local| superduper.misc.eager:309  | Select: MongoOutputs(identifier='MongoOutputs-identifier-uuid-e6f9f631-95f6-424b-ac12-2bdaa18c1dab-table-documents-parts-find-x-1-y-1-z-1-builds-1-files-1-blobs-1-schema-1-outputs-a-b', uuid='e6f9f631-95f6-424b-ac12-2bdaa18c1dab', table='documents', parts=[('find', (\{\}, \{'x': 1, 'y': 1, 'z': 1, '_builds': 1, '_files': 1, '_blobs': 1, '_schema': 1\}), \{\}), ('outputs', ('a', 'b'), \{\})])
    2024-Jul-25 11:20:57.16| INFO     | zhouhaha-2.local| superduper.misc.eager:311  | Predict id: c
    2024-Jul-25 11:20:57.16| INFO     | zhouhaha-2.local| superduper.misc.eager:313  | Predict kwargs: \{\}
    2024-Jul-25 11:20:57.16| INFO     | zhouhaha-2.local| superduper.misc.eager:323  | Listener: Listener(identifier='c', uuid='c', upstream=None, plugins=None, key=('x', 'y', 'z', '_outputs.a', '_outputs.b'), model=ObjectModel(identifier='c', uuid='2ef9ed89-e1a8-4f41-8c54-90e27866d1b2', upstream=None, plugins=None, signature='*args,**kwargs', datatype=None, output_schema=None, flatten=False, model_update_kwargs=\{\}, predict_kwargs=\{\}, compute_kwargs=\{\}, validation=None, metric_values=\{\}, num_workers=0, object=\<function func_c at 0x10777fb50\>), select=MongoOutputs(identifier='MongoOutputs-identifier-uuid-e6f9f631-95f6-424b-ac12-2bdaa18c1dab-table-documents-parts-find-x-1-y-1-z-1-builds-1-files-1-blobs-1-schema-1-outputs-a-b', uuid='e6f9f631-95f6-424b-ac12-2bdaa18c1dab', table='documents', parts=[('find', (\{\}, \{'x': 1, 'y': 1, 'z': 1, '_builds': 1, '_files': 1, '_blobs': 1, '_schema': 1\}), \{\}), ('outputs', ('a', 'b'), \{\})]), predict_kwargs=\{\})
    2024-Jul-25 11:20:57.16| INFO     | zhouhaha-2.local| superduper.components.listener:83   | Requesting listener setup on CDC service
    2024-Jul-25 11:20:57.16| INFO     | zhouhaha-2.local| superduper.components.listener:93   | Skipping listener setup on CDC service since no URI is set
    2024-Jul-25 11:20:57.16| INFO     | zhouhaha-2.local| superduper.jobs.queue:104  | Declaring component listener/c
    2024-Jul-25 11:20:57.16| INFO     | zhouhaha-2.local| superduper.jobs.queue:178  | Running jobs for listener::c
    2024-Jul-25 11:20:57.16| INFO     | zhouhaha-2.local| superduper.backends.local.compute:64   | Submitting job. function:\<function method_job at 0x108f1fd90\>
    2024-Jul-25 11:20:57.16| INFO     | zhouhaha-2.local| superduper.components.model:678  | Requesting prediction in db - [c] with predict_id c
    
    2024-Jul-25 11:20:57.17| INFO     | zhouhaha-2.local| superduper.components.model:807  | Adding 3 model outputs to `db`
    2024-Jul-25 11:20:57.17| SUCCESS  | zhouhaha-2.local| superduper.backends.local.compute:70   | Job submitted on \<superduper.backends.local.compute.LocalComputeBackend object at 0x146f38ee0\>.  function:\<function method_job at 0x108f1fd90\> future:755ecfce-0ac1-4078-b2ca-445670d4d478

</pre>
</details>

```python
list(db["documents"].select().outputs("a", "b", "c").select().execute())
```

<details>
<summary>Outputs</summary>
<pre>
    [Document(\{'_id': ObjectId('66a1c49909d51b458635eda2'), '_outputs': \{'c': \{'x': 1, 'y': '2', 'z': Encodable(identifier='e2d1839ed1706f7d470d87f8c48a5584cafa5a12', uuid='6ec1ba5e-0b93-49a0-a362-1637fd933849', datatype=DataType(identifier='numpy-int64[3]', uuid='72357574-0c3e-4503-b0c7-e15a6595491d', upstream=None, plugins=None, encoder=\<superduper.ext.numpy.encoder.EncodeArray object at 0x146f805b0\>, decoder=\<superduper.ext.numpy.encoder.DecodeArray object at 0x146f820b0\>, info=None, shape=[3], directory=None, encodable='encodable', bytes_encoding=\<BytesEncoding.BYTES: 'Bytes'\>, intermediate_type='bytes', media_type=None), uri=None, x=array([1, 2, 3])), 'o_a': \{'x': 1, 'model': 'a'\}, 'o_b': \{'x': 1, 'y': '2', 'o_a': \{'x': 1, 'model': 'a'\}, 'model': 'b'\}, 'model': 'c'\}, 'b': \{'x': 1, 'y': '2', 'o_a': \{'x': 1, 'model': 'a'\}, 'model': 'b'\}, 'a': \{'x': 1, 'model': 'a'\}\}\}),
     Document(\{'_id': ObjectId('66a1c49909d51b458635eda3'), '_outputs': \{'c': \{'x': 2, 'y': '3', 'z': Encodable(identifier='168d0ba38783ef14943e28073c750dbd4b83bdcc', uuid='9288eb18-1811-492c-a32e-28e467d31939', datatype=DataType(identifier='numpy-int64[3]', uuid='695b4336-8d43-4113-8b2c-70898ad94103', upstream=None, plugins=None, encoder=\<superduper.ext.numpy.encoder.EncodeArray object at 0x146fa3610\>, decoder=\<superduper.ext.numpy.encoder.DecodeArray object at 0x146fa2ad0\>, info=None, shape=[3], directory=None, encodable='encodable', bytes_encoding=\<BytesEncoding.BYTES: 'Bytes'\>, intermediate_type='bytes', media_type=None), uri=None, x=array([4, 5, 6])), 'o_a': \{'x': 2, 'model': 'a'\}, 'o_b': \{'x': 2, 'y': '3', 'o_a': \{'x': 2, 'model': 'a'\}, 'model': 'b'\}, 'model': 'c'\}, 'b': \{'x': 2, 'y': '3', 'o_a': \{'x': 2, 'model': 'a'\}, 'model': 'b'\}, 'a': \{'x': 2, 'model': 'a'\}\}\}),
     Document(\{'_id': ObjectId('66a1c49909d51b458635eda4'), '_outputs': \{'c': \{'x': 3, 'y': '4', 'z': Encodable(identifier='31c74b4f993d8a50e16d055f45359504dad76c05', uuid='e9a3d7dc-2d68-4110-8938-3f79955f871a', datatype=DataType(identifier='numpy-int64[3]', uuid='5049de2e-33d2-401b-9e1a-8b5a1bb352e8', upstream=None, plugins=None, encoder=\<superduper.ext.numpy.encoder.EncodeArray object at 0x146f94ac0\>, decoder=\<superduper.ext.numpy.encoder.DecodeArray object at 0x146f95180\>, info=None, shape=[3], directory=None, encodable='encodable', bytes_encoding=\<BytesEncoding.BYTES: 'Bytes'\>, intermediate_type='bytes', media_type=None), uri=None, x=array([7, 8, 9])), 'o_a': \{'x': 3, 'model': 'a'\}, 'o_b': \{'x': 3, 'y': '4', 'o_a': \{'x': 3, 'model': 'a'\}, 'model': 'b'\}, 'model': 'c'\}, 'b': \{'x': 3, 'y': '4', 'o_a': \{'x': 3, 'model': 'a'\}, 'model': 'b'\}, 'a': \{'x': 3, 'model': 'a'\}\}\})]
</pre>
</details>

If you want to modify the predict_id of a specific model, 
you can use `output.predict_id = "your_predict_id"` to set it.

```python
model_predict_id = ObjectModel(identifier="c", object=func_c)
output_predict_id = model_predict_id(data["x"], data["y"], data["z"], output_a, output_b)
output_predict_id.predict_id = "new_predict_id"
output_predict_id.apply()
```

<details>
<summary>Outputs</summary>
<pre>
    2024-Jul-25 11:20:57.21| INFO     | zhouhaha-2.local| superduper.misc.eager:289  | Applying nodes: ['a', 'b', 'new_predict_id']
    2024-Jul-25 11:20:57.21| INFO     | zhouhaha-2.local| superduper.misc.eager:298  | Node [a] already applied.
    2024-Jul-25 11:20:57.21| INFO     | zhouhaha-2.local| superduper.misc.eager:298  | Node [b] already applied.
    2024-Jul-25 11:20:57.21| INFO     | zhouhaha-2.local| superduper.misc.eager:301  | Applying node: new_predict_id
    2024-Jul-25 11:20:57.21| INFO     | zhouhaha-2.local| superduper.misc.eager:302  | Example output \{'x': 1, 'y': '2', 'z': array([1, 2, 3]), 'o_a': \{'x': 1, 'model': 'a'\}, 'o_b': \{'x': 1, 'y': '2', 'o_a': \{'x': 1, 'model': 'a'\}, 'model': 'b'\}, 'model': 'c'\}
    2024-Jul-25 11:20:57.21| INFO     | zhouhaha-2.local| superduper.misc.eager:306  | Key: ('x', 'y', 'z', '_outputs.a', '_outputs.b')
    2024-Jul-25 11:20:57.22| INFO     | zhouhaha-2.local| superduper.misc.eager:309  | Select: MongoOutputs(identifier='MongoOutputs-identifier-uuid-0e7da996-a427-43bd-8ad2-c2688da2c4b2-table-documents-parts-find-x-1-y-1-z-1-builds-1-files-1-blobs-1-schema-1-outputs-a-b', uuid='0e7da996-a427-43bd-8ad2-c2688da2c4b2', table='documents', parts=[('find', (\{\}, \{'x': 1, 'y': 1, 'z': 1, '_builds': 1, '_files': 1, '_blobs': 1, '_schema': 1\}), \{\}), ('outputs', ('a', 'b'), \{\})])
    2024-Jul-25 11:20:57.22| INFO     | zhouhaha-2.local| superduper.misc.eager:311  | Predict id: new_predict_id
    2024-Jul-25 11:20:57.22| INFO     | zhouhaha-2.local| superduper.misc.eager:313  | Predict kwargs: \{\}
    2024-Jul-25 11:20:57.22| INFO     | zhouhaha-2.local| superduper.misc.eager:323  | Listener: Listener(identifier='new_predict_id', uuid='new_predict_id', upstream=None, plugins=None, key=('x', 'y', 'z', '_outputs.a', '_outputs.b'), model=ObjectModel(identifier='c', uuid='f1623135-5b78-4528-b4c8-15fcecfa306a', upstream=None, plugins=None, signature='*args,**kwargs', datatype=None, output_schema=None, flatten=False, model_update_kwargs=\{\}, predict_kwargs=\{\}, compute_kwargs=\{\}, validation=None, metric_values=\{\}, num_workers=0, object=\<function func_c at 0x10777fb50\>), select=MongoOutputs(identifier='MongoOutputs-identifier-uuid-0e7da996-a427-43bd-8ad2-c2688da2c4b2-table-documents-parts-find-x-1-y-1-z-1-builds-1-files-1-blobs-1-schema-1-outputs-a-b', uuid='0e7da996-a427-43bd-8ad2-c2688da2c4b2', table='documents', parts=[('find', (\{\}, \{'x': 1, 'y': 1, 'z': 1, '_builds': 1, '_files': 1, '_blobs': 1, '_schema': 1\}), \{\}), ('outputs', ('a', 'b'), \{\})]), predict_kwargs=\{\})
    2024-Jul-25 11:20:57.22| WARNING  | zhouhaha-2.local| superduper.backends.local.artifacts:82   | File /tmp/test/7a3674bbe73b9998a81ee8daa6c12516bcac30ff already exists
    2024-Jul-25 11:20:57.22| INFO     | zhouhaha-2.local| superduper.components.listener:83   | Requesting listener setup on CDC service
    2024-Jul-25 11:20:57.22| INFO     | zhouhaha-2.local| superduper.components.listener:93   | Skipping listener setup on CDC service since no URI is set
    2024-Jul-25 11:20:57.22| INFO     | zhouhaha-2.local| superduper.jobs.queue:104  | Declaring component listener/new_predict_id
    2024-Jul-25 11:20:57.22| INFO     | zhouhaha-2.local| superduper.jobs.queue:178  | Running jobs for listener::new_predict_id
    2024-Jul-25 11:20:57.22| INFO     | zhouhaha-2.local| superduper.backends.local.compute:64   | Submitting job. function:\<function method_job at 0x108f1fd90\>
    2024-Jul-25 11:20:57.22| INFO     | zhouhaha-2.local| superduper.components.model:678  | Requesting prediction in db - [c] with predict_id new_predict_id
    
    2024-Jul-25 11:20:57.23| INFO     | zhouhaha-2.local| superduper.components.model:807  | Adding 3 model outputs to `db`
    2024-Jul-25 11:20:57.23| SUCCESS  | zhouhaha-2.local| superduper.backends.local.compute:70   | Job submitted on \<superduper.backends.local.compute.LocalComputeBackend object at 0x146f38ee0\>.  function:\<function method_job at 0x108f1fd90\> future:1ae144f7-db9f-41eb-95ab-1f4af75860f8

</pre>
</details>

View the prediction results of all data in the database.

```python
list(db["_outputs.new_predict_id"].select().execute())
```

<details>
<summary>Outputs</summary>
<pre>
    [Document(\{'_outputs': \{'new_predict_id': \{'x': 1, 'y': '2', 'z': Encodable(identifier='e2d1839ed1706f7d470d87f8c48a5584cafa5a12', uuid='1c2c16b6-66cf-414d-a0f7-e8702df24569', datatype=DataType(identifier='numpy-int64[3]', uuid='565196cb-fc5e-41f5-a293-8bb6db908046', upstream=None, plugins=None, encoder=\<superduper.ext.numpy.encoder.EncodeArray object at 0x146f80910\>, decoder=\<superduper.ext.numpy.encoder.DecodeArray object at 0x146f81240\>, info=None, shape=[3], directory=None, encodable='encodable', bytes_encoding=\<BytesEncoding.BYTES: 'Bytes'\>, intermediate_type='bytes', media_type=None), uri=None, x=array([1, 2, 3])), 'o_a': \{'x': 1, 'model': 'a'\}, 'o_b': \{'x': 1, 'y': '2', 'o_a': \{'x': 1, 'model': 'a'\}, 'model': 'b'\}, 'model': 'c'\}\}, '_source': ObjectId('66a1c49909d51b458635eda2'), '_fold': 'train', '_id': ObjectId('66a1c49909d51b458635edc0')\}),
     Document(\{'_outputs': \{'new_predict_id': \{'x': 2, 'y': '3', 'z': Encodable(identifier='168d0ba38783ef14943e28073c750dbd4b83bdcc', uuid='ccec3051-bc09-47d2-98f6-3209d51b197a', datatype=DataType(identifier='numpy-int64[3]', uuid='d7fb6e08-8c1f-4dd7-b277-541f5181af94', upstream=None, plugins=None, encoder=\<superduper.ext.numpy.encoder.EncodeArray object at 0x146f80b20\>, decoder=\<superduper.ext.numpy.encoder.DecodeArray object at 0x146f80c70\>, info=None, shape=[3], directory=None, encodable='encodable', bytes_encoding=\<BytesEncoding.BYTES: 'Bytes'\>, intermediate_type='bytes', media_type=None), uri=None, x=array([4, 5, 6])), 'o_a': \{'x': 2, 'model': 'a'\}, 'o_b': \{'x': 2, 'y': '3', 'o_a': \{'x': 2, 'model': 'a'\}, 'model': 'b'\}, 'model': 'c'\}\}, '_source': ObjectId('66a1c49909d51b458635eda3'), '_fold': 'train', '_id': ObjectId('66a1c49909d51b458635edc1')\}),
     Document(\{'_outputs': \{'new_predict_id': \{'x': 3, 'y': '4', 'z': Encodable(identifier='31c74b4f993d8a50e16d055f45359504dad76c05', uuid='d9fbb133-3d27-4433-8f7e-3dc4cd869db5', datatype=DataType(identifier='numpy-int64[3]', uuid='8f979f90-84d9-4ad6-92dc-e077b0767173', upstream=None, plugins=None, encoder=\<superduper.ext.numpy.encoder.EncodeArray object at 0x146f81a50\>, decoder=\<superduper.ext.numpy.encoder.DecodeArray object at 0x146f83d60\>, info=None, shape=[3], directory=None, encodable='encodable', bytes_encoding=\<BytesEncoding.BYTES: 'Bytes'\>, intermediate_type='bytes', media_type=None), uri=None, x=array([7, 8, 9])), 'o_a': \{'x': 3, 'model': 'a'\}, 'o_b': \{'x': 3, 'y': '4', 'o_a': \{'x': 3, 'model': 'a'\}, 'model': 'b'\}, 'model': 'c'\}\}, '_source': ObjectId('66a1c49909d51b458635eda4'), '_fold': 'train', '_id': ObjectId('66a1c49909d51b458635edc2')\})]
</pre>
</details>

If you want to perform if-like conditional operations to route data using different models, you can use `set_condition` to handle it. Currently, only equals and not equals conditions are supported.

```python
model_condition = ObjectModel(identifier="condition", object=func_a)
output_condition = model_condition(data["x"])
output_condition.set_condition(data["x"] == 1)
output_condition.apply()
output_condition
```

<details>
<summary>Outputs</summary>
<pre>
    2024-Jul-25 11:20:57.27| INFO     | zhouhaha-2.local| superduper.misc.eager:289  | Applying nodes: ['condition']
    2024-Jul-25 11:20:57.27| INFO     | zhouhaha-2.local| superduper.misc.eager:301  | Applying node: condition
    2024-Jul-25 11:20:57.27| INFO     | zhouhaha-2.local| superduper.misc.eager:302  | Example output \{'x': 1, 'model': 'a'\}
    2024-Jul-25 11:20:57.27| INFO     | zhouhaha-2.local| superduper.misc.eager:306  | Key: x
    2024-Jul-25 11:20:57.27| INFO     | zhouhaha-2.local| superduper.misc.eager:309  | Select: documents.find(\{'x': 1\}, \{'x': 1, '_builds': 1, '_files': 1, '_blobs': 1, '_schema': 1\})
    2024-Jul-25 11:20:57.27| INFO     | zhouhaha-2.local| superduper.misc.eager:311  | Predict id: condition
    2024-Jul-25 11:20:57.27| INFO     | zhouhaha-2.local| superduper.misc.eager:313  | Predict kwargs: \{\}
    2024-Jul-25 11:20:57.27| INFO     | zhouhaha-2.local| superduper.misc.eager:323  | Listener: Listener(identifier='condition', uuid='condition', upstream=None, plugins=None, key='x', model=ObjectModel(identifier='condition', uuid='b4e6eba0-cf23-4fc8-921b-2067d0950267', upstream=None, plugins=None, signature='*args,**kwargs', datatype=None, output_schema=None, flatten=False, model_update_kwargs=\{\}, predict_kwargs=\{\}, compute_kwargs=\{\}, validation=None, metric_values=\{\}, num_workers=0, object=\<function func_a at 0x146fa7d90\>), select=documents.find(\{'x': 1\}, \{'x': 1, '_builds': 1, '_files': 1, '_blobs': 1, '_schema': 1\}), predict_kwargs=\{\})
    2024-Jul-25 11:20:57.27| WARNING  | zhouhaha-2.local| superduper.backends.local.artifacts:82   | File /tmp/test/d24f9039a299416f14abfa8a11830e67a5665b98 already exists
    2024-Jul-25 11:20:57.27| INFO     | zhouhaha-2.local| superduper.components.listener:83   | Requesting listener setup on CDC service
    2024-Jul-25 11:20:57.27| INFO     | zhouhaha-2.local| superduper.components.listener:93   | Skipping listener setup on CDC service since no URI is set
    2024-Jul-25 11:20:57.27| INFO     | zhouhaha-2.local| superduper.jobs.queue:104  | Declaring component listener/condition
    2024-Jul-25 11:20:57.28| INFO     | zhouhaha-2.local| superduper.jobs.queue:178  | Running jobs for listener::condition
    2024-Jul-25 11:20:57.28| INFO     | zhouhaha-2.local| superduper.backends.local.compute:64   | Submitting job. function:\<function method_job at 0x108f1fd90\>
    2024-Jul-25 11:20:57.28| INFO     | zhouhaha-2.local| superduper.components.model:678  | Requesting prediction in db - [condition] with predict_id condition
    
    2024-Jul-25 11:20:57.28| INFO     | zhouhaha-2.local| superduper.components.model:807  | Adding 1 model outputs to `db`
    2024-Jul-25 11:20:57.28| SUCCESS  | zhouhaha-2.local| superduper.backends.local.compute:70   | Job submitted on \<superduper.backends.local.compute.LocalComputeBackend object at 0x146f38ee0\>.  function:\<function method_job at 0x108f1fd90\> future:b432ac85-9c9f-4ec3-bb50-02cdbe09830f

</pre>
<pre>
    \{'x': 1, 'model': 'a'\}
</pre>
</details>

```python
db['documents'].find({}, {'x': 1, '_builds': 1, '_files': 1, '_blobs': 1, '_schema': 1}).filter({'x': 1}).execute()
```

<details>
<summary>Outputs</summary>
<pre>
    SuperDuperCursor(raw_cursor=\<mongomock.collection.Cursor object at 0x146f826b0\>, id_field='_id', db=\<superduper.base.datalayer.Datalayer object at 0x146f38f70\>, scores=None, schema=None, process_func=None, _it=0)
</pre>
</details>

```python
list(db["_outputs.condition"].select().execute())
```

<details>
<summary>Outputs</summary>
<pre>
    [Document(\{'_outputs': \{'condition': \{'x': 1, 'model': 'a'\}\}, '_source': ObjectId('66a1c49909d51b458635eda2'), '_fold': 'train', '_id': ObjectId('66a1c49909d51b458635edc7')\})]
</pre>
</details>
