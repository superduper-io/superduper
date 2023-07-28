from __future__ import annotations

import typing as t
from contextlib import contextmanager

if t.TYPE_CHECKING:
    from torch import device as _device

    from .wrapper import TorchModel


def device_of(model: TorchModel) -> t.Union[_device, str]:
    """
    Get device of a model.

    :param model: PyTorch model
    """
    try:
        return next(iter(model.state_dict().values())).device
    except StopIteration:
        return 'cpu'


@contextmanager
def eval(model: TorchModel) -> t.Iterator[None]:
    """
    Temporarily set a model to evaluation mode.

    :param model: PyTorch model
    """
    was_training = model.training  # type: ignore[attr-defined]
    try:
        model.eval()
        yield
    finally:
        if was_training:
            model.train()


@contextmanager
def set_device(model: TorchModel, device: _device):
    """
    Temporarily set a device of a model.

    :param model: PyTorch model
    :param device: Device to set
    """
    device_before = device_of(model)
    try:
        model.to(device)  # type: ignore[attr-defined]
        yield
    finally:
        model.to(device_before)  # type: ignore[attr-defined]


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
