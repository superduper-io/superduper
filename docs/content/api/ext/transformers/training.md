**`superduper.ext.transformers.training`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/ext/transformers/training.py)

## `create_quantization_config` 

```python
create_quantization_config(config: superduper.ext.transformers.training.LLMTrainer)
```
| Parameter | Description |
|-----------|-------------|
| config | The configuration to use. |

Create quantization config for LLM training.

## `handle_ray_results` 

```python
handle_ray_results(db,
     llm,
     results)
```
| Parameter | Description |
|-----------|-------------|
| db | datalayer, used for saving the checkpoint |
| llm | llm model, used for saving the checkpoint |
| results | the ray training results, contains the checkpoint |

Handle the ray results.

Will save the checkpoint to db if db and llm provided.

## `prepare_lora_training` 

```python
prepare_lora_training(model,
     config: superduper.ext.transformers.training.LLMTrainer)
```
| Parameter | Description |
|-----------|-------------|
| model | The model to prepare for LoRA training. |
| config | The configuration to use. |

Prepare LoRA training for the model.

Get the LoRA target modules and convert the model to peft model.

## `train_func` 

```python
train_func(training_args: superduper.ext.transformers.training.LLMTrainer,
     train_dataset: 'Dataset',
     eval_datasets: Union[ForwardRef('Dataset'),
     Dict[str,
     ForwardRef('Dataset')]],
     model_kwargs: dict,
     tokenizer_kwargs: dict,
     trainer_prepare_func: Optional[Callable] = None,
     callbacks=None,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| training_args | training Arguments, see LLMTrainingArguments |
| train_dataset | training dataset, can be huggingface datasets.Dataset or ray.data.Dataset |
| eval_datasets | evaluation dataset, can be a dict of datasets |
| model_kwargs | model kwargs for AutoModelForCausalLM |
| tokenizer_kwargs | tokenizer kwargs for AutoTokenizer |
| trainer_prepare_func | function to prepare trainer This function will be called after the trainer is created, we can add some custom settings to the trainer |
| callbacks | list of callbacks will be added to the trainer |
| kwargs | other kwargs for Trainer All the kwargs will be passed to Trainer, make sure the Trainer support these kwargs |

Base training function for LLM model.

## `tokenize` 

```python
tokenize(tokenizer,
     example,
     X,
     y)
```
| Parameter | Description |
|-----------|-------------|
| tokenizer | The tokenizer to use. |
| example | The example to tokenize. |
| X | The input key. |
| y | The output key. |

Function to tokenize the example.

## `train` 

```python
train(training_args: superduper.ext.transformers.training.LLMTrainer,
     train_dataset: datasets.arrow_dataset.Dataset,
     eval_datasets: Union[datasets.arrow_dataset.Dataset,
     Dict[str,
     datasets.arrow_dataset.Dataset]],
     model_kwargs: dict,
     tokenizer_kwargs: dict,
     db: Optional[ForwardRef('Datalayer')] = None,
     llm: Optional[ForwardRef('LLM')] = None,
     ray_configs: Optional[dict] = None,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| training_args | training Arguments, see LLMTrainingArguments |
| train_dataset | training dataset |
| eval_datasets | evaluation dataset, can be a dict of datasets |
| model_kwargs | model kwargs for AutoModelForCausalLM |
| tokenizer_kwargs | tokenizer kwargs for AutoTokenizer |
| db | datalayer, used for creating LLMCallback |
| llm | llm model, used for creating LLMCallback |
| ray_configs | ray configs, must provide if using ray |
| kwargs | other kwargs for Trainer |

Train LLM model on specified dataset.

The training process can be run on these following modes:
- Local node without ray, but only support single GPU
- Local node with ray, support multi-nodes and multi-GPUs
- Remote node with ray, support multi-nodes and multi-GPUs

If run locally, will use train_func to train the model.
Can log the training process to db if db and llm provided.
Will reuse the db and llm from the current process.
If run on ray, will use ray_train to train the model.
Can log the training process to db if db and llm provided.
Will rebuild the db and llm for the new process that can access to db.
The ray cluster must can access to db.

## `Checkpoint` 

```python
Checkpoint(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = None,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None,
     path: Optional[str],
     step: int) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| artifacts | A dictionary of artifacts paths and `DataType` objects |
| path | The path to the checkpoint. |
| step | The step of the checkpoint. |

Checkpoint component for saving the model checkpoint.

## `LLMCallback` 

```python
LLMCallback(self,
     cfg: Optional[ForwardRef('Config')] = None,
     identifier: Optional[str] = None,
     db: Optional[ForwardRef('Datalayer')] = None,
     llm: Optional[ForwardRef('LLM')] = None,
     experiment_id: Optional[str] = None)
```
| Parameter | Description |
|-----------|-------------|
| cfg | The configuration to use. |
| identifier | The identifier to use. |
| db | The datalayer to use. |
| llm | The LLM model to use. |
| experiment_id | The experiment id to use. |

LLM Callback for logging training process to db.

This callback will save the checkpoint to db after each epoch.
If the save_total_limit is set, will remove the oldest checkpoint.

