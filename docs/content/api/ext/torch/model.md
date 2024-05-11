**`superduperdb.ext.torch.model`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/ext/torch/model.py)

## `create_batch` 

```python
create_batch(args)
```
| Parameter | Description |
|-----------|-------------|
| args | single data point for batching |

Create a singleton batch in a manner similar to the PyTorch dataloader.

```python
create_batch(3.).shape
# torch.Size([1])
x, y = create_batch([torch.randn(5), torch.randn(3, 7)])
x.shape
# torch.Size([1, 5])
y.shape
# torch.Size([1, 3, 7])
d = create_batch(({'a': torch.randn(4)}))
d['a'].shape
# torch.Size([1, 4])
```

## `torchmodel` 

```python
torchmodel(class_obj)
```
| Parameter | Description |
|-----------|-------------|
| class_obj | Class to decorate |

A decorator to convert a `torch.nn.Module` into a `TorchModel`.

Decorate a `torch.nn.Module` so that when it is invoked,
the result is a `TorchModel`.

## `unpack_batch` 

```python
unpack_batch(args)
```
| Parameter | Description |
|-----------|-------------|
| args | a batch of model outputs |

Unpack a batch into lines of tensor output.

```python
unpack_batch(torch.randn(1, 10))[0].shape
# torch.Size([10])
out = unpack_batch([torch.randn(2, 10), torch.randn(2, 3, 5)])
type(out)
# <class 'list'>
len(out)
# 2
out = unpack_batch({'a': torch.randn(2, 10), 'b': torch.randn(2, 3, 5)})
[type(x) for x in out]
# [<class 'dict'>, <class 'dict'>]
out[0]['a'].shape
# torch.Size([10])
out[0]['b'].shape
# torch.Size([3, 5])
out = unpack_batch({'a': {'b': torch.randn(2, 10)}})
out[0]['a']['b'].shape
# torch.Size([10])
out[1]['a']['b'].shape
# torch.Size([10])
```

## `TorchModel` 

```python
TorchModel(self,
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
     object: 'torch.nn.Module',
     preprocess: 't.Optional[t.Callable]' = None,
     preprocess_signature: 'Signature' = 'singleton',
     postprocess: 't.Optional[t.Callable]' = None,
     postprocess_signature: 'Signature' = 'singleton',
     forward_method: 'str' = '__call__',
     forward_signature: 'Signature' = 'singleton',
     train_forward_method: 'str' = '__call__',
     train_forward_signature: 'Signature' = 'singleton',
     train_preprocess: 't.Optional[t.Callable]' = None,
     train_preprocess_signature: 'Signature' = 'singleton',
     collate_fn: 't.Optional[t.Callable]' = None,
     optimizer_state: 't.Optional[t.Any]' = None,
     loader_kwargs: 't.Dict' = <factory>) -> None
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
| object | Torch model, e.g. `torch.nn.Module` |
| preprocess | Preprocess function, the function to apply to the input |
| preprocess_signature | The signature of the preprocess function |
| postprocess | The postprocess function, the function to apply to the output |
| postprocess_signature | The signature of the postprocess function |
| forward_method | The forward method, the method to call on the model |
| forward_signature | The signature of the forward method |
| train_forward_method | Train forward method, the method to call on the model |
| train_forward_signature | The signature of the train forward method |
| train_preprocess | Train preprocess function, the function to apply to the input |
| train_preprocess_signature | The signature of the train preprocess function |
| collate_fn | The collate function for the dataloader |
| optimizer_state | The optimizer state |
| loader_kwargs | The kwargs for the dataloader |
| trainer | `Trainer` object to train the model |
| preferred_devices | The order of devices to use |
| device | The device to be used |

Torch model. This class is a wrapper around a PyTorch model.

## `BasicDataset` 

```python
BasicDataset(self,
     items,
     transform,
     signature)
```
| Parameter | Description |
|-----------|-------------|
| items | items, typically documents |
| transform | function, typically a preprocess function |
| signature | signature of the transform function |

Basic database iterating over a list of documents and applying a transformation.

