**`superduperdb.ext.utils`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/ext/utils.py)

## `str_shape` 

```python
str_shape(shape: Sequence[int]) -> str
```
| Parameter | Description |
|-----------|-------------|
| shape | The shape to convert. |

Convert a shape to a string.

## `format_prompt` 

```python
format_prompt(X: str,
     prompt: str,
     context: Optional[List[str]] = None) -> str
```
| Parameter | Description |
|-----------|-------------|
| X | The input to format the prompt with. |
| prompt | The prompt to format. |
| context | The context to format the prompt with. |

Format a prompt with the given input and context.

## `get_key` 

```python
get_key(key_name: str) -> str
```
| Parameter | Description |
|-----------|-------------|
| key_name | The name of the environment variable to get. |

Get an environment variable.

## `superduperdecode` 

```python
superduperdecode(r: Any,
     encoders: Union[Dict[str,
     ForwardRef('DataType')],
     ForwardRef('LoadDict')])
```
| Parameter | Description |
|-----------|-------------|
| r | The object to decode. |
| encoders | The encoders to use. |

Decode a superduper encoded object.

## `superduperencode` 

```python
superduperencode(object)
```
| Parameter | Description |
|-----------|-------------|
| object | The object to encode. |

Encode an object using superduper.

