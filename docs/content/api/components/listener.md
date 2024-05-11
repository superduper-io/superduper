**`superduperdb.components.listener`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/components/listener.py)

## `Listener` 

```python
Listener(self,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     identifier: str = '',
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     key: Union[str,
     List[str],
     Tuple[List[str],
     Dict[str,
     str]]],
     model: superduperdb.components.model.Model,
     select: superduperdb.backends.base.query.Query,
     active: bool = True,
     predict_kwargs: Optional[Dict] = <factory>) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | A string used to identify the model. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| artifacts | A dictionary of artifacts paths and `DataType` objects |
| key | Key to be bound to the model. |
| model | Model for processing data. |
| select | Object for selecting which data is processed. |
| active | Toggle to ``False`` to deactivate change data triggering. |
| predict_kwargs | Keyword arguments to self.model.predict(). |

Listener component.

Listener object which is used to process a column/key of a collection or table,
and store the outputs.

