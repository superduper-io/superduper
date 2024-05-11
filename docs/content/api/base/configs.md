**`superduperdb.base.configs`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/base/configs.py)

## `ConfigError` 

```python
ConfigError(self,
     /,
     *args,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| args | *args for `Exception` |
| kwargs | **kwargs for `Exception` |

An exception raised when there is an error in the configuration.

## `ConfigSettings` 

```python
ConfigSettings(self,
     cls: Type,
     environ: Optional[Dict] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| cls | The Pydantic class to read. |
| environ | The environment variables to read from. |

Helper class to read a configuration from a dataclass.

Reads a dataclass class from a configuration file and environment variables.

