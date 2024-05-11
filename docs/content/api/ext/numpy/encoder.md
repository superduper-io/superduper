**`superduperdb.ext.numpy.encoder`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/ext/numpy/encoder.py)

## `array` 

```python
array(dtype: str,
     shape: Sequence,
     bytes_encoding: Optional[str] = None,
     encodable: str = 'encodable')
```
| Parameter | Description |
|-----------|-------------|
| dtype | The dtype of the array. |
| shape | The shape of the array. |
| bytes_encoding | The bytes encoding to use. |
| encodable | The encodable to use. |

Create an encoder of numpy arrays.

## `DecodeArray` 

```python
DecodeArray(self,
     dtype,
     shape)
```
| Parameter | Description |
|-----------|-------------|
| dtype | The dtype of the array. |
| shape | The shape of the array. |

Decode a numpy array from bytes.

## `EncodeArray` 

```python
EncodeArray(self,
     dtype)
```
| Parameter | Description |
|-----------|-------------|
| dtype | The dtype of the array. |

Encode a numpy array to bytes.

