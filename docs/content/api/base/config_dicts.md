**`superduper.base.config_dicts`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/base/config_dicts.py)

## `combine_configs` 

```python
combine_configs(dicts: Sequence[Dict[str,
     object]]) -> Dict[str,
     object]
```
| Parameter | Description |
|-----------|-------------|
| dicts | The dictionaries to combine. |

Combine a sequence of dictionaries into a single dictionary.

## `environ_to_config_dict` 

```python
environ_to_config_dict(prefix: str,
     parent: Dict[str,
     str],
     environ: Optional[Dict[str,
     str]] = None,
     err: Optional[TextIO] = <_io.TextIOWrapper name='<stderr>' mode='w' encoding='utf-8'>,
     fail: bool = False)
```
| Parameter | Description |
|-----------|-------------|
| prefix | The prefix to use for environment variables. |
| parent | The parent dictionary to use as a basis. |
| environ | The environment variables to read from. |
| err | The file to write errors to. |
| fail | Whether to raise an exception on error. |

Convert environment variables to a configuration dictionary.

