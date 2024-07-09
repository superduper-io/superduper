**`superduper.ext.cohere.model`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/ext/cohere/model.py)

## `CohereEmbed` 

```python
CohereEmbed(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = None,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     signature: str = 'singleton',
     datatype: 'EncoderArg' = None,
     output_schema: 't.Optional[Schema]' = None,
     flatten: 'bool' = False,
     model_update_kwargs: 't.Dict' = None,
     predict_kwargs: 't.Dict' = None,
     compute_kwargs: 't.Dict' = None,
     validation: 't.Optional[Validation]' = None,
     metric_values: 't.Dict' = None,
     model: 't.Optional[str]' = None,
     max_batch_size: 'int' = 8,
     client_kwargs: Dict[str,
     Any] = None,
     shape: Optional[Sequence[int]] = None,
     batch_size: int = 100) -> None
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
| client_kwargs | The keyword arguments to pass to the client. |
| shape | The shape as ``tuple`` of the embedding. |
| batch_size | The batch size to use for the predictor. |

Cohere embedding predictor.

## `CohereGenerate` 

```python
CohereGenerate(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = None,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     signature: str = '*args,
    **kwargs',
     datatype: 'EncoderArg' = None,
     output_schema: 't.Optional[Schema]' = None,
     flatten: 'bool' = False,
     model_update_kwargs: 't.Dict' = None,
     predict_kwargs: 't.Dict' = None,
     compute_kwargs: 't.Dict' = None,
     validation: 't.Optional[Validation]' = None,
     metric_values: 't.Dict' = None,
     model: 't.Optional[str]' = None,
     max_batch_size: 'int' = 8,
     client_kwargs: Dict[str,
     Any] = None,
     takes_context: bool = True,
     prompt: str = '') -> None
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
| client_kwargs | The keyword arguments to pass to the client. |
| takes_context | Whether the model takes context into account. |
| prompt | The prompt to use to seed the response. |

Cohere realistic text generator (chat predictor).

## `Cohere` 

```python
Cohere(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = None,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     signature: 'Signature' = '*args,
    **kwargs',
     datatype: 'EncoderArg' = None,
     output_schema: 't.Optional[Schema]' = None,
     flatten: 'bool' = False,
     model_update_kwargs: 't.Dict' = None,
     predict_kwargs: 't.Dict' = None,
     compute_kwargs: 't.Dict' = None,
     validation: 't.Optional[Validation]' = None,
     metric_values: 't.Dict' = None,
     model: 't.Optional[str]' = None,
     max_batch_size: 'int' = 8,
     client_kwargs: Dict[str,
     Any] = None) -> None
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
| client_kwargs | The keyword arguments to pass to the client. |

Cohere predictor.

