from contextlib import contextmanager
import typing as t

import torch
from torch import nn


def device_of(model: nn.Module) -> str:
    """
    Get device of a model.

    :param model: PyTorch model
    """
    try:
        return next(iter(model.state_dict().values())).device
    except StopIteration:
        return 'cpu'


@contextmanager
def eval(model: nn.Module) -> None:
    was_training = model.training
    try:
        model.eval()
        yield
    finally:
        if was_training:
            model.train()


@contextmanager
def set_device(model: nn.Module, device: torch.device) -> None:
    """
    Temporarily set a device of a model.

    :param model: PyTorch model
    :param device: Device to set
    """
    device_before = device_of(model)
    try:
        model.to(device)
        yield
    finally:
        model.to(device_before)


def to_device(
    item: t.Union[t.Tuple, t.List, t.Dict, torch.Tensor], device: torch.device
) -> t.Union[t.Dict, torch.Tensor]:
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
    return item.to(device)
