**`superduper.components.template`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper.components/template.py)

## `Template` 

```python
Template(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = None,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     component: Union[superduper.components.component.Component,
     Dict],
     info: Optional[Dict] = None,
     _component_blobs: Union[Dict,
     bytes,
     NoneType] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| artifacts | A dictionary of artifacts paths and `DataType` objects |
| component | Template component with variables. |
| info | Info. |
| _component_blobs | Blobs in `Template.component` NOTE: This is only for internal use. |

Application template component.

