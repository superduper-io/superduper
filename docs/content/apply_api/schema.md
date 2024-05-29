# `Schema`

- Apply a dictionary of `FieldType` and `DataType` to encode columnar data
- Mostly relevant to SQL databases, but can also be used with MongoDB
- `Schema` leverages encoding functionality of contained `DataType` instances

***Dependencies***

- [`DataType`](./datatype.md)

***Usage pattern***

(Learn how to build a `DataType` [here](datatype))

*Vanilla usage*

Table can potentially include
more columns which don't need encoding:

```python
from superduperdb import Schema

schema = Schema(
    'my-schema',
    fields={
        'img': dt_1,   # A `DataType`
        'video': dt_2,   # Another `DataType`
    }
)

db.apply(schema)
```

*Usage with SQL*

All columns should be flagged with either `DataType` or `dtype`:

```python
from superduperdb.backends.ibis import dtype

schema = Schema(
    'my-schema',
    fields={
        'img': dt_1,   # A `DataType`
        'video': dt_2,   # Another `DataType`
        'txt', dtype('str'),
        'numer', dtype('int'),
    }
)

db.apply(schema)
```

*Usage with MongoDB*

In MongoDB, the non-`DataType` columns/ fields can be omitted:

```python
schema = Schema(
    'my-schema',
    fields={
        'img': dt_1,   # A `DataType`
        'video': dt_2,   # Another `DataType`
    }
)

db.apply(schema)
```

*Usage with `Model` descendants (MongoDB only)*

If used together with `Model`, the model is assumed to emit `tuple` outputs, and these 
need differential encoding. The `Schema` is applied to the columns of output, 
to get something which can be saved in the `db.databackend`.

```python
from superduperdb import ObjectModel

m = Model(
    'my-model',
    object=my_object,
    output_schema=schema
)

db.apply(m)    # adds model and schema
```

***See also***

- [Change-data capture](../cluster_mode/change_data_capture)