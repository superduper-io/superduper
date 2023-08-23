from __future__ import annotations

import typing as t
from contextlib import contextmanager

if t.TYPE_CHECKING:
    from torch import device as _device
    from torch.nn.modules import Module


def device_of(module: Module) -> t.Union[_device, str]:
    """
    Get device of a model.

    :param model: PyTorch model
    """
    try:
        return next(iter(module.state_dict().values())).device
    except StopIteration:
        return 'cpu'


@contextmanager
def eval(module: Module) -> t.Iterator[None]:
    """
    Temporarily set a module to evaluation mode.

    :param module: PyTorch module
    """
    was_training = module.training
    try:
        module.eval()
        yield
    finally:
        if was_training:
            module.train()


@contextmanager
def set_device(module: Module, device: _device):
    """
    Temporarily set a device of a module.

    :param module: PyTorch module
    :param device: Device to set
    """
    device_before = device_of(module)
    try:
        module.to(device)
        yield
    finally:
        module.to(device_before)


def to_device(
    item: t.Any,  # lists or dicts of Tensors
    device: t.Union[str, _device],
) -> t.Any:
    """
    Send tensor leaves of nested list/ dictionaries/ tensors to device.

    :param item: torch.Tensor instance
    :param device: device to which one would like to send
    """
    if isinstance(item, tuple):
        item = list(item)
    if isinstance(item, list):
        for i, it in enumerate(item):
            item[i] = to_device(it, device)
        return item
    if isinstance(item, dict):
        for k in item:
            item[k] = to_device(item[k], device)
        return item
    if hasattr(item, 'to'):
        return item.to(device)
    return item
