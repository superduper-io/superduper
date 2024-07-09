**`superduper.base.code`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/base/code.py)

## `Code` 

```python
Code(self,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = None,
     *,
     identifier: str = '',
     code: str) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| code | The code to store. |

A class to store remote code.

This class stores remote code that can be executed on a remote server.

