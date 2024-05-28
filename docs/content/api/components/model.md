**`superduperdb.components.model`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/components/model.py)

## `codemodel` 

```python
codemodel(item: 't.Optional[t.Callable]' = None,
     identifier: 't.Optional[str]' = None,
     datatype=None,
     model_update_kwargs: 't.Optional[t.Dict]' = None,
     flatten: 'bool' = False,
     output_schema: 't.Optional[Schema]' = None)
```
| Parameter | Description |
|-----------|-------------|
| item | Callable to wrap with `CodeModel`. |
| identifier | Identifier for the `CodeModel`. |
| datatype | Datatype for the model outputs. |
| model_update_kwargs | Dictionary to define update kwargs. |
| flatten | If `True`, flatten the outputs and save. |
| output_schema | Schema for the model outputs. |

Decorator to wrap a function with `CodeModel`.

When a function is wrapped with this decorator,
the function comes out as a `CodeModel`.

## `objectmodel` 

```python
objectmodel(item: 't.Optional[t.Callable]' = None,
     identifier: 't.Optional[str]' = None,
     datatype=None,
     model_update_kwargs: 't.Optional[t.Dict]' = None,
     flatten: 'bool' = False,
     output_schema: 't.Optional[Schema]' = None)
```
| Parameter | Description |
|-----------|-------------|
| item | Callable to wrap with `ObjectModel`. |
| identifier | Identifier for the `ObjectModel`. |
| datatype | Datatype for the model outputs. |
| model_update_kwargs | Dictionary to define update kwargs. |
| flatten | If `True`, flatten the outputs and save. |
| output_schema | Schema for the model outputs. |

Decorator to wrap a function with `ObjectModel`.

When a function is wrapped with this decorator,
the function comes out as an `ObjectModel`.

## `CodeModel` 

```python
CodeModel(self,
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
     num_workers: 'int' = 0,
     object: 'Code') -> None
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
| num_workers | Number of workers to use for parallel processing |
| object | Code object |

Model component which stores a code object.

## `Model` 

```python
Model(self,
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
     metric_values: 't.Dict' = <factory>) -> None
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

Base class for components which can predict.

## `ObjectModel` 

```python
ObjectModel(self,
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
     num_workers: 'int' = 0,
     object: 't.Any') -> None
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
| num_workers | Number of workers to use for parallel processing |
| object | Model/ computation object |

Model component which wraps a Model to become serializable.

```python
# Example:
# -------
m = ObjectModel('test', lambda x: x + 2)
m.predict_one(2)
# 4
```

## `QueryModel` 

```python
QueryModel(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     signature: 'Signature' = '**kwargs',
     datatype: 'EncoderArg' = None,
     output_schema: 't.Optional[Schema]' = None,
     flatten: 'bool' = False,
     model_update_kwargs: 't.Dict' = <factory>,
     predict_kwargs: 't.Dict' = <factory>,
     compute_kwargs: 't.Dict' = <factory>,
     validation: 't.Optional[Validation]' = None,
     metric_values: 't.Dict' = <factory>,
     preprocess: 't.Optional[t.Callable]' = None,
     postprocess: 't.Optional[t.Union[t.Callable]]' = None,
     select: 'Query') -> None
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
| preprocess | Preprocess callable |
| postprocess | Postprocess callable |
| select | query used to find data (can include `like`) |

QueryModel component.

Model which can be used to query data and return those
precomputed queries as Results.

## `Validation` 

```python
Validation(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     metrics: 't.Sequence[Metric]' = (),
     key: 't.Optional[ModelInputType]' = None,
     datasets: 't.Sequence[Dataset]' = ()) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| artifacts | A dictionary of artifacts paths and `DataType` objects |
| metrics | List of metrics for validation |
| key | Model input type key |
| datasets | Sequence of dataset. |

component which represents Validation definition.

## `Mapping` 

```python
Mapping(self,
     mapping: 'ModelInputType',
     signature: 'Signature')
```
| Parameter | Description |
|-----------|-------------|
| mapping | Mapping that represents a collection or table map. |
| signature | Signature for the model. |

Class to represent model inputs for mapping database collections or tables.

## `APIBaseModel` 

```python
APIBaseModel(self,
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
     max_batch_size: 'int' = 8) -> None
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

APIBaseModel component which is used to make the type of API request.

## `APIModel` 

```python
APIModel(self,
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
     url: 'str',
     postprocess: 't.Optional[t.Callable]' = None) -> None
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
| url | The url to use for the API request |
| postprocess | Postprocess function to use on the output of the API request |

APIModel component which is used to make the type of API request.

## `CallableInputs` 

```python
CallableInputs(self,
     fn,
     predict_kwargs: 't.Dict' = {})
```
| Parameter | Description |
|-----------|-------------|
| fn | Callable function |
| predict_kwargs | (optional) predict_kwargs if provided in Model initiation |

Class represents the model callable args and kwargs.

## `IndexableNode` 

```python
IndexableNode(self,
     types: 't.Sequence[t.Type]') -> None
```
| Parameter | Description |
|-----------|-------------|
| types | Sequence of types |

Base indexable node for `ObjectModel`.

## `Inputs` 

```python
Inputs(self,
     params)
```
| Parameter | Description |
|-----------|-------------|
| params | List of parameters of the Model object |

Base class to represent the model args and kwargs.

## `SequentialModel` 

```python
SequentialModel(self,
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
     models: 't.List[Model]') -> None
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
| models | A list of models to use |

Sequential model component which wraps a model to become serializable.

## `Trainer` 

```python
Trainer(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     key: 'ModelInputType',
     select: 'Query',
     transform: 't.Optional[t.Callable]' = None,
     metric_values: 't.Dict' = <factory>,
     signature: 'Signature' = '*args',
     data_prefetch: 'bool' = False,
     prefetch_size: 'int' = 1000,
     prefetch_factor: 'int' = 100,
     in_memory: 'bool' = True,
     compute_kwargs: 't.Dict' = <factory>) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| artifacts | A dictionary of artifacts paths and `DataType` objects |
| key | Model input type key. |
| select | Model select query for training. |
| transform | (optional) transform callable. |
| metric_values | Dictionary for metric defaults. |
| signature | Model signature. |
| data_prefetch | Boolean for prefetching data before forward pass. |
| prefetch_size | Prefetch batch size. |
| prefetch_factor | Prefetch factor for data prefetching. |
| in_memory | If training in memory. |
| compute_kwargs | Kwargs for compute backend. |

Trainer component to train a model.

Training configuration object, containing all settings necessary for a particular
learning task use-case to be serialized and initiated. The object is ``callable``
and returns a class which may be invoked to apply training.

