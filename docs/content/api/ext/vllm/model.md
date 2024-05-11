**`superduperdb.ext.vllm.model`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/ext/vllm/model.py)

## `VllmAPI` 

```python
VllmAPI(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     api_url: str = '',
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
     max_batch_size: Optional[int] = 4) -> None
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
| api_url | The URL for the API. |

Wrapper for requesting the vLLM API service.

API Server format, started by `vllm.entrypoints.api_server`.

## `VllmModel` 

```python
VllmModel(self,
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
     model_name: str = '',
     tensor_parallel_size: int = 1,
     trust_remote_code: bool = True,
     vllm_kwargs: dict = <factory>,
     on_ray: bool = False,
     ray_address: Optional[str] = None,
     ray_config: dict = <factory>) -> None
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
| model_name | The name of the model to use. |
| tensor_parallel_size | The number of tensor parallelism. |
| trust_remote_code | Whether to trust remote code. |
| vllm_kwargs | Additional arguments to pass to the VLLM |
| on_ray | Whether to use Ray for parallelism. |
| ray_address | The address of the Ray cluster. |
| ray_config | The configuration for Ray. |

Load a large language model from VLLM.

