**`superduperdb.ext.llamacpp.model`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/ext/llamacpp/model.py)

## `download_uri` 

```python
download_uri(uri,
     save_path)
```
| Parameter | Description |
|-----------|-------------|
| uri | URI to download |
| save_path | place to save |

Download file.

## `LlamaCpp` 

```python
LlamaCpp(self,
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
     prompt: str = '{input}',
     prompt_func: Optional[Callable] = None,
     max_batch_size: Optional[int] = 4,
     model_name_or_path: str = 'facebook/opt-125m',
     model_kwargs: Dict = <factory>,
     download_dir: str = '.llama_cpp') -> None
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
| prompt | The template to use for the prompt. |
| prompt_func | The function to use for the prompt. |
| max_batch_size | The maximum batch size to use for batch generation. |
| model_name_or_path | path or name of model |
| model_kwargs | dictionary of init-kwargs |
| download_dir | local caching directory |

Llama.cpp connector.

## `LlamaCppEmbedding` 

```python
LlamaCppEmbedding(self,
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
     prompt: str = '{input}',
     prompt_func: Optional[Callable] = None,
     max_batch_size: Optional[int] = 4,
     model_name_or_path: str = 'facebook/opt-125m',
     model_kwargs: Dict = <factory>,
     download_dir: str = '.llama_cpp') -> None
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
| prompt | The template to use for the prompt. |
| prompt_func | The function to use for the prompt. |
| max_batch_size | The maximum batch size to use for batch generation. |
| model_name_or_path | path or name of model |
| model_kwargs | dictionary of init-kwargs |
| download_dir | local caching directory |

Llama.cpp connector for embeddings.

