**`superduperdb.ext.sentence_transformers.model`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/ext/sentence_transformers/model.py)

## `SentenceTransformer` 

```python
SentenceTransformer(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     preferred_devices: 't.Sequence[str]' = ('cuda',
     'mps',
     'cpu'),
     device: str = 'cpu',
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     signature: Literal['*args',
     '**kwargs',
     '*args,
    **kwargs',
     'singleton'] = 'singleton',
     datatype: 'EncoderArg' = None,
     output_schema: 't.Optional[Schema]' = None,
     flatten: 'bool' = False,
     model_update_kwargs: 't.Dict' = <factory>,
     predict_kwargs: 't.Dict' = <factory>,
     compute_kwargs: 't.Dict' = <factory>,
     validation: 't.Optional[Validation]' = None,
     metric_values: 't.Dict' = <factory>,
     object: Optional[sentence_transformers.SentenceTransformer.SentenceTransformer] = None,
     model: Optional[str] = None,
     preprocess: Optional[Callable] = None,
     postprocess: Optional[Callable] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| artifacts | A dictionary of artifacts paths and `DataType` objects |
| signature | The signature of the model. |
| datatype | DataType instance. |
| output_schema | Output schema (mapping of encoders). |
| flatten | Flatten the model outputs. |
| model_update_kwargs | The kwargs to use for model update. |
| predict_kwargs | Additional arguments to use at prediction time. |
| compute_kwargs | Kwargs used for compute backend job submit. Example (Ray backend): compute_kwargs = dict(resources=...). |
| validation | The validation ``Dataset`` instances to use. |
| metric_values | The metrics to evaluate on. |
| object | The SentenceTransformer object to use. |
| model | The model name, e.g. 'all-MiniLM-L6-v2'. |
| device | The device to use, e.g. 'cpu' or 'cuda'. |
| preprocess | The preprocessing function to apply to the input. |
| postprocess | The postprocessing function to apply to the output. |
| preferred_devices | A list of devices to prefer, in that order. |

A model for sentence embeddings using `sentence-transformers`.

