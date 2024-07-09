**`superduper.ext.torch.encoder`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/ext/torch/encoder.py)

## `tensor` 

```python
tensor(dtype,
     shape: Sequence,
     bytes_encoding: Optional[str] = None,
     encodable: str = 'encodable',
     db: Optional[ForwardRef('Datalayer')] = None)
```
| Parameter | Description |
|-----------|-------------|
| dtype | The dtype of the tensor. |
| shape | The shape of the tensor. |
| bytes_encoding | The bytes encoding to use. |
| encodable | The encodable name ["artifact", "encodable", "lazy_artifact", "file"]. |
| db | The datalayer instance. |

Create an encoder for a tensor of a given dtype and shape.

## `DecodeTensor` 

```python
DecodeTensor(self,
     dtype,
     shape)
```
| Parameter | Description |
|-----------|-------------|
| dtype | The dtype of the tensor, eg. torch.float32 |
| shape | The shape of the tensor, eg. (3, 4) |

Decode a tensor from bytes.

## `EncodeTensor` 

```python
EncodeTensor(self,
     dtype)
```
| Parameter | Description |
|-----------|-------------|
| dtype | The dtype of the tensor, eg. torch.float32 |

Encode a tensor to bytes.

