**`superduperdb.base.variables`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/base/variables.py)

## `Variable` 

```python
Variable(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     setter_callback: dataclasses.InitVar[typing.Optional[typing.Callable]] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| setter_callback | A callback function that takes the value, datalayer and kwargs as input and returns the formatted variable. |

Mechanism for allowing "free variables" in a leaf object.

The idea is to allow a variable to be set at runtime, rather than
at object creation time.

## `VariableError` 

```python
VariableError(self,
     /,
     *args,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| args | *args for `Exception`. |
| kwargs | **kwargs for `Exception`. |

Variable error.

