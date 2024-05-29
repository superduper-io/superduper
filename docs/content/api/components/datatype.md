**`superduperdb.components.datatype`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/components/datatype.py)

## `pickle_decode` 

```python
pickle_decode(b: bytes,
     info: Optional[Dict] = None) -> Any
```
| Parameter | Description |
|-----------|-------------|
| b | The bytes to decode. |
| info | Optional information. |

Decodes bytes using pickle.

## `pickle_encode` 

```python
pickle_encode(object: Any,
     info: Optional[Dict] = None) -> bytes
```
| Parameter | Description |
|-----------|-------------|
| object | The object to encode. |
| info | Optional information. |

Encodes an object using pickle.

## `base64_to_bytes` 

```python
base64_to_bytes(encoded)
```
| Parameter | Description |
|-----------|-------------|
| encoded | The base64 encoded string. |

Decodes a base64 encoded string.

## `bytes_to_base64` 

```python
bytes_to_base64(bytes)
```
| Parameter | Description |
|-----------|-------------|
| bytes | The bytes to convert. |

Converts bytes to base64.

## `dill_decode` 

```python
dill_decode(b: bytes,
     info: Optional[Dict] = None) -> Any
```
| Parameter | Description |
|-----------|-------------|
| b | The bytes to decode. |
| info | Optional information. |

Decodes bytes using dill.

## `dill_encode` 

```python
dill_encode(object: Any,
     info: Optional[Dict] = None) -> bytes
```
| Parameter | Description |
|-----------|-------------|
| object | The object to encode. |
| info | Optional information. |

Encodes an object using dill.

## `encode_torch_state_dict` 

```python
encode_torch_state_dict(module,
     info)
```
| Parameter | Description |
|-----------|-------------|
| module | Module. |
| info | Information. |

Encode torch state dictionary.

## `file_check` 

```python
file_check(path: Any,
     info: Optional[Dict] = None) -> str
```
| Parameter | Description |
|-----------|-------------|
| path | The file path to check. |
| info | Optional information. |

Checks if a file path exists.

## `get_serializer` 

```python
get_serializer(identifier: str,
     method: str,
     encodable: str,
     db: Optional[ForwardRef('Datalayer')] = None)
```
| Parameter | Description |
|-----------|-------------|
| identifier | The identifier of the serializer. |
| method | The method of the serializer. |
| encodable | The type of encodable object. |
| db | The Datalayer instance. |

Get a serializer.

## `json_decode` 

```python
json_decode(b: str,
     info: Optional[Dict] = None) -> Any
```
| Parameter | Description |
|-----------|-------------|
| b | The JSON string to decode |
| info | Optional information |

Decode the JSON string to an dict.

## `json_encode` 

```python
json_encode(object: Any,
     info: Optional[Dict] = None) -> str
```
| Parameter | Description |
|-----------|-------------|
| object | The object to encode |
| info | Optional information |

Encode the dict to a JSON string.

## `torch_decode` 

```python
torch_decode(b: bytes,
     info: Optional[Dict] = None) -> Any
```
| Parameter | Description |
|-----------|-------------|
| b | The bytes to decode. |
| info | Optional information. |

Decodes bytes to a torch model.

## `torch_encode` 

```python
torch_encode(object: Any,
     info: Optional[Dict] = None) -> bytes
```
| Parameter | Description |
|-----------|-------------|
| object | The object to encode. |
| info | Optional information. |

Saves an object in torch format.

## `Encoder` 

```python
Encoder(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     encoder: Optional[Callable] = None,
     decoder: Optional[Callable] = None,
     info: Optional[Dict] = None,
     shape: Optional[Sequence] = None,
     directory: Optional[str] = None,
     encodable: str = 'encodable',
     bytes_encoding: Optional[str] = <BytesEncoding.BYTES: 'Bytes'>,
     intermediate_type: Optional[str] = 'bytes',
     media_type: Optional[str] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| artifacts | A dictionary of artifacts paths and `DataType` objects |
| encoder | A callable that converts an encodable object of this encoder to bytes. |
| decoder | A callable that converts bytes to an encodable object of this encoder. |
| info | An optional information dictionary. |
| shape | The shape of the data. |
| directory | The directory to store file types. |
| encodable | The type of encodable object ('encodable', 'lazy_artifact', or 'file'). |
| bytes_encoding | The encoding type for bytes ('base64' or 'bytes'). |
| intermediate_type | Type of the intermediate data [IntermediateType.BYTES, IntermediateType.STRING] |
| media_type | The media type. |

A data type component that defines how data is encoded and decoded.

## `Artifact` 

```python
Artifact(self,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     x: Any = <EMPTY>,
     *,
     identifier: str = '',
     file_id: Optional[str] = None,
     datatype: superduperdb.components.datatype.DataType,
     uri: Optional[str] = None,
     sha1: Optional[str] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| file_id | unique-id of the content |
| datatype | The datatype of the content. |
| uri | URI of the content, if any. |
| sha1 | SHA1 hash of the content. |
| x | The artifact object. |

Class for representing data to be saved on disk or in the artifact-store.

## `DecodeTorchStateDict` 

```python
DecodeTorchStateDict(self,
     cls)
```
| Parameter | Description |
|-----------|-------------|
| cls | Torch state cls |

Torch state dictionary decoder.

## `Encodable` 

```python
Encodable(self,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     x: Any = <EMPTY>,
     blob: dataclasses.InitVar[typing.Optional[bytearray]] = None,
     *,
     identifier: str = '',
     file_id: Optional[str] = None,
     datatype: superduperdb.components.datatype.DataType,
     uri: Optional[str] = None,
     sha1: Optional[str] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| file_id | unique-id of the content |
| datatype | The datatype of the content. |
| uri | URI of the content, if any. |
| sha1 | SHA1 hash of the content. |
| x | The encodable object. |
| blob | The blob data. |

Class for encoding non-Python datatypes to the database.

## `File` 

```python
File(self,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     x: Any = <EMPTY>,
     file_name: Optional[str] = None,
     *,
     identifier: str = '',
     file_id: Optional[str] = None,
     datatype: superduperdb.components.datatype.DataType,
     uri: Optional[str] = None,
     sha1: Optional[str] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| file_id | unique-id of the content |
| datatype | The datatype of the content. |
| uri | URI of the content, if any. |
| sha1 | SHA1 hash of the content. |
| x | path to the file |
| file_name | File name |

Data to be saved on disk and passed as a file reference.

## `LazyArtifact` 

```python
LazyArtifact(self,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     x: Any = <EMPTY>,
     *,
     identifier: str = '',
     file_id: Optional[str] = None,
     datatype: superduperdb.components.datatype.DataType,
     uri: Optional[str] = None,
     sha1: Optional[str] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| file_id | unique-id of the content |
| datatype | The datatype of the content. |
| uri | URI of the content, if any. |
| sha1 | SHA1 hash of the content. |
| x | The artifact object. |

Data to be saved and loaded only when needed.

## `LazyFile` 

```python
LazyFile(self,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     x: Any = <EMPTY>,
     file_name: Optional[str] = None,
     *,
     identifier: str = '',
     file_id: Optional[str] = None,
     datatype: superduperdb.components.datatype.DataType,
     uri: Optional[str] = None,
     sha1: Optional[str] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| file_id | unique-id of the content |
| datatype | The datatype of the content. |
| uri | URI of the content, if any. |
| sha1 | SHA1 hash of the content. |
| x | path to the file |
| file_name | File name |

Class is used to load a file only when needed.

## `Native` 

```python
Native(self,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     x: Optional[Any] = None,
     *,
     identifier: str = '',
     file_id: Optional[str] = None,
     datatype: superduperdb.components.datatype.DataType,
     uri: Optional[str] = None,
     sha1: Optional[str] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| file_id | unique-id of the content |
| datatype | The datatype of the content. |
| uri | URI of the content, if any. |
| sha1 | SHA1 hash of the content. |
| x | The encodable object. |

Class for representing native data supported by the underlying database.

