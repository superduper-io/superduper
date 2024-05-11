**`superduperdb.components.vector_index`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/components/vector_index.py)

## `sqlvector` 

```python
sqlvector(shape)
```
| Parameter | Description |
|-----------|-------------|
| shape | The shape of the vector |

Create an encoder for a vector (list of ints/ floats) of a given shape.

This is used for compatibility with SQL databases, as the default vector

## `vector` 

```python
vector(shape,
     identifier: Optional[str] = None)
```
| Parameter | Description |
|-----------|-------------|
| shape | The shape of the vector |
| identifier | The identifier of the vector |

Create an encoder for a vector (list of ints/ floats) of a given shape.

## `VectorIndex` 

```python
VectorIndex(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     indexing_listener: superduperdb.components.listener.Listener,
     compatible_listener: Optional[superduperdb.components.listener.Listener] = None,
     measure: superduperdb.vector_search.base.VectorIndexMeasureType = <VectorIndexMeasureType.cosine: 'cosine'>,
     metric_values: Optional[Dict] = <factory>) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| artifacts | A dictionary of artifacts paths and `DataType` objects |
| indexing_listener | Listener which is applied to created vectors |
| compatible_listener | Listener which is applied to vectors to be compared |
| measure | Measure to use for comparison |
| metric_values | Metric values for this index |

A component carrying the information to apply a vector index.

## `DecodeArray` 

```python
DecodeArray(self,
     dtype)
```
| Parameter | Description |
|-----------|-------------|
| dtype | Datatype of array |

Class to decode an array.

## `EncodeArray` 

```python
EncodeArray(self,
     dtype)
```
| Parameter | Description |
|-----------|-------------|
| dtype | Datatype of array |

Class to encode an array.

