**`superduper.components.stack`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper.components/stack.py)

## `Stack` 

```python
Stack(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = None,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     components: Sequence[superduper.components.component.Component]) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| artifacts | A dictionary of artifacts paths and `DataType` objects |
| components | List of components to stack together and add to database. |

A placeholder to hold list of components under a namespace.

