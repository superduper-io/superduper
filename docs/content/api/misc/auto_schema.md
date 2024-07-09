**`superduper.misc.auto_schema`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/misc/auto_schema.py)

## `infer_datatype` 

```python
infer_datatype(data: Any) -> Union[superduper.components.datatype.DataType,
     type,
     NoneType]
```
| Parameter | Description |
|-----------|-------------|
| data | The data object |

Infer the datatype of a given data object.

If the data object is a base type, return None,
Otherwise, return the inferred datatype

## `infer_schema` 

```python
infer_schema(data: Mapping[str,
     Any],
     identifier: Optional[str] = None,
     ibis=False) -> superduper.components.schema.Schema
```
| Parameter | Description |
|-----------|-------------|
| data | The data object |
| identifier | The identifier for the schema, if None, it will be generated |
| ibis | If True, the schema will be updated for the Ibis backend, otherwise for MongoDB |

Infer a schema from a given data object.

## `register_module` 

```python
register_module(module_name)
```
| Parameter | Description |
|-----------|-------------|
| module_name | The module name, e.g. "superduper.ext.numpy.encoder" |

Register a module for datatype inference.

Only modules with a check and create function will be registered

## `updated_schema_data_for_ibis` 

```python
updated_schema_data_for_ibis(schema_data) -> Dict[str,
     superduper.components.datatype.DataType]
```
| Parameter | Description |
|-----------|-------------|
| schema_data | The schema data |

Update the schema data for Ibis backend.

Convert the basic data types to Ibis data types.

## `updated_schema_data_for_mongodb` 

```python
updated_schema_data_for_mongodb(schema_data) -> Dict[str,
     superduper.components.datatype.DataType]
```
| Parameter | Description |
|-----------|-------------|
| schema_data | The schema data |

Update the schema data for MongoDB backend.

Only keep the data types that can be stored directly in MongoDB.

