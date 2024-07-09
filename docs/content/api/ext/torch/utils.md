**`superduper.ext.torch.utils`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/ext/torch/utils.py)

## `device_of` 

```python
device_of(module: 'Module') -> 't.Union[_device,
     str]'
```
| Parameter | Description |
|-----------|-------------|
| module | PyTorch model |

Get device of a model.

## `eval` 

```python
eval(module: 'Module') -> 't.Iterator[None]'
```
| Parameter | Description |
|-----------|-------------|
| module | PyTorch module |

Temporarily set a module to evaluation mode.

## `to_device` 

```python
to_device(item: 't.Any',
     device: 't.Union[str,
     _device]') -> 't.Any'
```
| Parameter | Description |
|-----------|-------------|
| item | torch.Tensor instance |
| device | device to which one would like to send |

Send tensor leaves of nested list/ dictionaries/ tensors to device.

## `set_device` 

```python
set_device(module: 'Module',
     device: '_device')
```
| Parameter | Description |
|-----------|-------------|
| module | PyTorch module |
| device | Device to set |

Temporarily set a device of a module.

