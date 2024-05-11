**`superduperdb.base.serializable`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/base/serializable.py)

## `Serializable` 

```python
Serializable(self) -> None
```
Base class for serializable objects.

This class is used to serialize and
deserialize objects to and from JSON + Artifact instances.

## `Variable` 

```python
Variable(self,
     value: Any,
     setter_callback: dataclasses.InitVar[typing.Optional[typing.Callable]] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| value | The name of the variable to be set at runtime. |
| setter_callback | A callback function that takes the value, datalayer and kwargs as input and returns the formatted variable. |

Mechanism for allowing "free variables" in a serializable object.

The idea is to allow a variable to be set at runtime, rather than
at object creation time.

## `VariableError` 

```python
VariableError(self,
     /,
     *args,
     **kwargs)
```
Variable error.

