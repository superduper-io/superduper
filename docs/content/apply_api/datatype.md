# `DataType`

- Convert objects which should be added to the database or model outputs to encoded `bytes`
- `DataType` encodings and decodings are fully configurable and can be written as functions
- Users may choose to encode `bytes` to strings with `base64` encoding

***Usage pattern***

Default `DataTypes`, called "serializers":

```python
from superduperdb.components.datatype import serializers

pickle_serializer = serializers['pickle']
```

Build your own `DataType` which saves data directly in the database:

```python
from superduperdb import DataType

dt = DataType(
    'my-datatype',
    encoder=function_from_object_to_bytes,
    decoder=function_from_bytes_to_object,
    encodable='encodable',
)

db.apply(dt)
```

:::info
*How do I choose the `encodable` parameter?*


| Value | Usage | 
| --- | --- |
| `"encodable"` | `dt` adds object encoded as `bytes` directly to the `db.databackend` |
| `"artifact"`  | `dt` saves object encoded as `bytes` to the `db.artifact_store` and a reference to `db.databackend` |
| `"lazy_artifact"` | as per `"artifact"` but `bytes` must be actively loaded when needed |
| `"file"` | `dt` simply saves a reference to a file in `db.artifact_store`; user handles loading |
:::

***See also***

- [Encoding-difficult data](../advanced_usage/encoding_difficult_data)