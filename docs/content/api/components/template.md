**`superduperdb.components.template`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/components/template.py)

## `Template` 

```python
Template(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     component: Union[superduperdb.components.component.Component,
     Dict],
     info: Optional[Dict] = <factory>,
     _component_blobs: Union[Dict,
     bytes,
     NoneType] = <factory>) -> None
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

