**`superduper.components.metric`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper.components/metric.py)

## `Metric` 

```python
Metric(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = None,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     object: Callable) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| artifacts | A dictionary of artifacts paths and `DataType` objects |
| object | Callable or an Artifact to be applied to the data. |

Metric base object used to evaluate performance on a dataset.

These objects are callable and are applied row-wise to the data, and averaged.

