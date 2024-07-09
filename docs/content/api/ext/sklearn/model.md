**`superduper.ext.sklearn.model`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/ext/sklearn/model.py)

## `Estimator` 

```python
Estimator(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = None,
     *,
     trainer: Optional[superduper.ext.sklearn.model.SklearnTrainer] = None,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     signature: Literal['*args',
     '**kwargs',
     '*args,
    **kwargs',
     'singleton'] = 'singleton',
     datatype: 'EncoderArg' = None,
     output_schema: 't.Optional[Schema]' = None,
     flatten: 'bool' = False,
     model_update_kwargs: 't.Dict' = None,
     predict_kwargs: 't.Dict' = None,
     compute_kwargs: 't.Dict' = None,
     validation: 't.Optional[Validation]' = None,
     metric_values: 't.Dict' = None,
     object: sklearn.base.BaseEstimator,
     preprocess: Optional[Callable] = None,
     postprocess: Optional[Callable] = None) -> None
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
| object | The estimator object from `sklearn`. |
| trainer | The trainer to use. |
| preprocess | The preprocessing function to use. |
| postprocess | The postprocessing function to use. |

Estimator model.

This is a model that can be trained and used for prediction.

## `SklearnTrainer` 

```python
SklearnTrainer(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = None,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     key: 'ModelInputType',
     select: 'Query',
     transform: 't.Optional[t.Callable]' = None,
     metric_values: 't.Dict' = None,
     signature: 'Signature' = '*args',
     data_prefetch: 'bool' = False,
     prefetch_size: 'int' = 1000,
     prefetch_factor: 'int' = 100,
     in_memory: 'bool' = True,
     compute_kwargs: 't.Dict' = None,
     fit_params: Dict = None,
     predict_params: Dict = None,
     y_preprocess: Optional[Callable] = None) -> None
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
| fit_params | The parameters to pass to `fit`. |
| predict_params | The parameters to pass to `predict |
| y_preprocess | The preprocessing function to use for the target. |

A trainer for `sklearn` models.

