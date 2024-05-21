**`superduperdb.ext.transformers.model`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/ext/transformers/model.py)

## `LLM` 

```python
LLM(self,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     trainer: 't.Optional[Trainer]' = None,
     identifier: str = '',
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
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
     model_name_or_path: Optional[str] = None,
     adapter_id: Union[str,
     superduperdb.ext.transformers.training.Checkpoint,
     NoneType] = None,
     model_kwargs: Dict = <factory>,
     tokenizer_kwargs: Dict = <factory>,
     prompt_template: str = '{input}') -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | model identifier |
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
| prompt_func | prompt function, default is None |
| max_batch_size | The maximum batch size to use for batch generation. |
| model_name_or_path | model name or path |
| adapter_id | adapter id, default is None Add a adapter to the base model for inference. |
| model_kwargs | model kwargs, all the kwargs will pass to `transformers.AutoModelForCausalLM.from_pretrained` |
| tokenizer_kwargs | tokenizer kwargs, all the kwargs will pass to `transformers.AutoTokenizer.from_pretrained` |
| prompt_template | prompt template, default is `"{input}"` |

LLM model based on `transformers` library.

All the `model_kwargs` will pass to
`transformers.AutoModelForCausalLM.from_pretrained`.
All the `tokenize_kwargs` will pass to
`transformers.AutoTokenizer.from_pretrained`.
When `model_name_or_path`, `bits`, `model_kwargs`, `tokenizer_kwargs` are the same,
will share the same base model and tokenizer cache.

## `TextClassificationPipeline` 

```python
TextClassificationPipeline(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = <factory>,
     *,
     preferred_devices: 't.Sequence[str]' = ('cuda',
     'mps',
     'cpu'),
     device: 't.Optional[str]' = None,
     trainer: 't.Optional[Trainer]' = None,
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
     tokenizer_name: Optional[str] = None,
     tokenizer_cls: object = <class 'transformers.models.auto.tokenization_auto.AutoTokenizer'>,
     tokenizer_kwargs: Dict = <factory>,
     model_name: Optional[str] = None,
     model_cls: object = <class 'transformers.models.auto.modeling_auto.AutoModelForSequenceClassification'>,
     model_kwargs: Dict = <factory>,
     pipeline: Optional[transformers.pipelines.base.Pipeline] = None,
     task: str = 'text-classification') -> None
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
| tokenizer_name | tokenizer name |
| tokenizer_cls | tokenizer class, e.g. ``transformers.AutoTokenizer`` |
| tokenizer_kwargs | tokenizer kwargs, will pass to ``tokenizer_cls`` |
| model_name | model name, will pass to ``model_cls`` |
| model_cls | model class, e.g. ``AutoModelForSequenceClassification`` |
| model_kwargs | model kwargs, will pass to ``model_cls`` |
| pipeline | pipeline instance, default is None, will build when None |
| task | task of the pipeline |
| trainer | `TransformersTrainer` instance |
| preferred_devices | preferred devices |
| device | device to use |

A wrapper for ``transformers.Pipeline``.

```python
# Example:
# -------
model = TextClassificationPipeline(...)
```

