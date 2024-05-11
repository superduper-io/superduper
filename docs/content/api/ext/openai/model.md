**`superduperdb.ext.openai.model`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/ext/openai/model.py)

## `OpenAIChatCompletion` 

```python
OpenAIChatCompletion(self,
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
     openai_api_key: Optional[str] = None,
     openai_api_base: Optional[str] = None,
     client_kwargs: Optional[dict] = <factory>,
     batch_size: int = 1,
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
| openai_api_key | The OpenAI API key. |
| openai_api_base | The server to use for requests. |
| client_kwargs | The kwargs to be passed to OpenAI |
| batch_size | The batch size to use. |
| prompt | The prompt to use to seed the response. |

OpenAI chat completion predictor.

## `OpenAIEmbedding` 

```python
OpenAIEmbedding(self,
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
     openai_api_key: Optional[str] = None,
     openai_api_base: Optional[str] = None,
     client_kwargs: Optional[dict] = <factory>,
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
| openai_api_key | The OpenAI API key. |
| openai_api_base | The server to use for requests. |
| client_kwargs | The kwargs to be passed to OpenAI |
| shape | The shape as ``tuple`` of the embedding. |
| batch_size | The batch size to use. |

OpenAI embedding predictor.

## `OpenAIAudioTranscription` 

```python
OpenAIAudioTranscription(self,
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
     openai_api_key: Optional[str] = None,
     openai_api_base: Optional[str] = None,
     client_kwargs: Optional[dict] = <factory>,
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
| openai_api_key | The OpenAI API key. |
| openai_api_base | The server to use for requests. |
| client_kwargs | The kwargs to be passed to OpenAI |
| takes_context | Whether the model takes context into account. |
| prompt | The prompt to guide the model's style. |

OpenAI audio transcription predictor.

The prompt should contain the `"context"` format variable.

## `OpenAIAudioTranslation` 

```python
OpenAIAudioTranslation(self,
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
     openai_api_key: Optional[str] = None,
     openai_api_base: Optional[str] = None,
     client_kwargs: Optional[dict] = <factory>,
     takes_context: bool = True,
     prompt: str = '',
     batch_size: int = 1) -> None
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
| openai_api_key | The OpenAI API key. |
| openai_api_base | The server to use for requests. |
| client_kwargs | The kwargs to be passed to OpenAI |
| takes_context | Whether the model takes context into account. |
| prompt | The prompt to guide the model's style. |
| batch_size | The batch size to use. |

OpenAI audio translation predictor.

The prompt should contain the `"context"` format variable.

## `OpenAIImageCreation` 

```python
OpenAIImageCreation(self,
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
     openai_api_key: Optional[str] = None,
     openai_api_base: Optional[str] = None,
     client_kwargs: Optional[dict] = <factory>,
     takes_context: bool = True,
     prompt: str = '',
     n: int = 1,
     response_format: str = 'b64_json') -> None
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
| openai_api_key | The OpenAI API key. |
| openai_api_base | The server to use for requests. |
| client_kwargs | The kwargs to be passed to OpenAI |
| takes_context | Whether the model takes context into account. |
| prompt | The prompt to use to seed the response. |
| n | The number of images to generate. |
| response_format | The response format to use. |

OpenAI image creation predictor.

## `OpenAIImageEdit` 

```python
OpenAIImageEdit(self,
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
     openai_api_key: Optional[str] = None,
     openai_api_base: Optional[str] = None,
     client_kwargs: Optional[dict] = <factory>,
     takes_context: bool = True,
     prompt: str = '',
     response_format: str = 'b64_json',
     n: int = 1) -> None
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
| openai_api_key | The OpenAI API key. |
| openai_api_base | The server to use for requests. |
| client_kwargs | The kwargs to be passed to OpenAI |
| takes_context | Whether the model takes context into account. |
| prompt | The prompt to use to seed the response. |
| response_format | The response format to use. |
| n | The number of images to generate. |

OpenAI image edit predictor.

