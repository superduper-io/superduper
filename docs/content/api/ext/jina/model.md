**`superduperdb.ext.jina.model`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/ext/jina/model.py)

## `JinaEmbedding` 

```python
JinaEmbedding(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     signature: str = 'singleton',
     datatype: 'EncoderArg' = None,
     output_schema: 't.Optional[Schema]' = None,
     flatten: 'bool' = False,
     model_update_kwargs: 't.Dict' = <factory>,
     predict_kwargs: 't.Dict' = <factory>,
     compute_kwargs: 't.Dict' = <factory>,
     validation: 't.Optional[Validation]' = None,
     metric_values: 't.Dict' = <factory>,
     model: 't.Optional[str]' = None,
     max_batch_size: 'int' = 8,
     api_key: Optional[str] = None,
     batch_size: int = 100,
     shape: Optional[Sequence[int]] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| artifacts | A dictionary of artifacts paths and `DataType` objects |
| signature | Model signature. |
| datatype | DataType instance. |
| output_schema | Output schema (mapping of encoders). |
| flatten | Flatten the model outputs. |
| model_update_kwargs | The kwargs to use for model update. |
| predict_kwargs | Additional arguments to use at prediction time. |
| compute_kwargs | Kwargs used for compute backend job submit. Example (Ray backend): compute_kwargs = dict(resources=...). |
| validation | The validation ``Dataset`` instances to use. |
| metric_values | The metrics to evaluate on. |
| model | The Model to use, e.g. ``'text-embedding-ada-002'`` |
| max_batch_size | Maximum  batch size. |
| api_key | The API key to use for the predicto |
| batch_size | The batch size to use for the predictor. |
| shape | The shape of the embedding as ``tuple``. If not provided, it will be obtained by sending a simple query to the API |

Jina embedding predictor.

## `Jina` 

```python
Jina(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     signature: 'Signature' = '*args,
    **kwargs',
     datatype: 'EncoderArg' = None,
     output_schema: 't.Optional[Schema]' = None,
     flatten: 'bool' = False,
     model_update_kwargs: 't.Dict' = <factory>,
     predict_kwargs: 't.Dict' = <factory>,
     compute_kwargs: 't.Dict' = <factory>,
     validation: 't.Optional[Validation]' = None,
     metric_values: 't.Dict' = <factory>,
     model: 't.Optional[str]' = None,
     max_batch_size: 'int' = 8,
     api_key: Optional[str] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| artifacts | A dictionary of artifacts paths and `DataType` objects |
| signature | Model signature. |
| datatype | DataType instance. |
| output_schema | Output schema (mapping of encoders). |
| flatten | Flatten the model outputs. |
| model_update_kwargs | The kwargs to use for model update. |
| predict_kwargs | Additional arguments to use at prediction time. |
| compute_kwargs | Kwargs used for compute backend job submit. Example (Ray backend): compute_kwargs = dict(resources=...). |
| validation | The validation ``Dataset`` instances to use. |
| metric_values | The metrics to evaluate on. |
| model | The Model to use, e.g. ``'text-embedding-ada-002'`` |
| max_batch_size | Maximum  batch size. |
| api_key | The API key to use for the predicto |

Cohere predictor.

