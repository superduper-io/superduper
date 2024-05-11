**`superduperdb.ext.pillow.encoder`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/ext/pillow/encoder.py)

## `encode_pil_image` 

```python
encode_pil_image(x,
     info: Optional[Dict] = None)
```
| Parameter | Description |
|-----------|-------------|
| x | The image to encode. |
| info | Additional information. |

Encode a `PIL.Image` to bytes.

## `image_type` 

```python
image_type(identifier: str,
     encodable: str = 'lazy_artifact',
     media_type: str = 'image/png',
     db: Optional[ForwardRef('Datalayer')] = None)
```
| Parameter | Description |
|-----------|-------------|
| identifier | The identifier for the data type. |
| encodable | The encodable type. |
| media_type | The media type. |
| db | The datalayer instance. |

Create a `DataType` for an image.

## `DecoderPILImage` 

```python
DecoderPILImage(self,
     handle_exceptions: bool = True)
```
| Parameter | Description |
|-----------|-------------|
| handle_exceptions | return a blank image if failure |

Decoder to convert `bytes` back into a `PIL.Image` class.

