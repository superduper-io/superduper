import torch

from superduperdb.training.loading import BasicDataset
from superduperdb.utils import create_batch, unpack_batch, progressbar


class Container(torch.nn.Module):
    def __init__(self, preprocessor=None, forward=None, postprocessor=None):
        super().__init__()
        self._preprocess = preprocessor
        self._forward = forward
        self._postprocess = postprocessor

    def preprocess(self, *args, **kwargs):
        if self._preprocess is not None:
            return self._preprocess(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    def postprocess(self, *args, **kwargs):
        if self._postprocess is not None:
            return self._postprocess(*args, **kwargs)


class TrivialContainer:
    def __init__(self, preprocess=None):
        self.preprocess = preprocess


def create_container(preprocess=None, forward=None, postprocess=None):
    if forward is not None:
        assert isinstance(forward, torch.nn.Module)
    if postprocess is not None:
        assert forward is not None
    if forward is not None:
        return TrivialContainer(preprocess)
    return Container(preprocessor=preprocess, forward=forward, postprocessor=postprocess)


def apply_model(model, args, single=True, **kwargs):
    """
    Apply model to args including pre-processing, forward pass and post-processing.

    :param model: model object including methods *preprocess*, *forward* and *postprocess*
    :param args: single or multiple data points over which to evaluate model
    :param single: toggle to apply model to single or multiple (batched) datapoints.
    :param kwargs: key, value pairs to be passed to dataloader

    >>> from types import SimpleNamespace
    >>> model = SimpleNamespace(
    ...     preprocess=lambda x: x + 2,
    ...     forward=torch.nn.Linear(16, 4),
    ...     postprocess=lambda x: x.topk(1)[1].item()
    ... )
    >>> out = apply_model(model, torch.randn(16))
    >>> isinstance(out, int)
    True
    >>> out = apply_model(model, [torch.randn(16), torch.randn(16)], single=False, batch_size=2)
    >>> isinstance(out, list)
    True
    >>> len(out)
    2

    >>> model = SimpleNamespace(
    ...     preprocess=lambda x: x + 2,
    ...     postprocess=lambda x: torch.cat([x, x])
    ... )
    >>> out = apply_model(model, torch.randn(16))
    >>> out.shape
    torch.Size([32])
    >>> out = apply_model(model, [torch.randn(16), torch.randn(16)], single=False)
    >>> out[0].shape
    torch.Size([32])

    >>> model = SimpleNamespace(
    ...     forward=torch.nn.Linear(16, 4),
    ...     postprocess=lambda x: x.topk(1)[1].item()
    ... )
    >>> out = apply_model(model, [torch.randn(16), torch.randn(16)], single=False, batch_size=2)

    """
    if single:
        if hasattr(model, 'preprocess'):
            args = model.preprocess(args)
        if hasattr(model, 'forward'):
            singleton_batch = create_batch(args)
            output = model.forward(singleton_batch)
            args = unpack_batch(output)[0]
        if hasattr(model, 'postprocess'):
            args = model.postprocess(args)
        return args
    else:
        if hasattr(model, 'preprocess'):
            inputs = BasicDataset(args, model.preprocess)
            loader = torch.utils.data.DataLoader(inputs, **kwargs)
        else:
            loader = torch.utils.data.DataLoader(args, **kwargs)
        out = []
        for batch in progressbar(loader, total=len(loader)):
            if hasattr(model, 'forward'):
                tmp = model.forward(batch)
                tmp = unpack_batch(tmp)
            else:
                tmp = unpack_batch(batch)
            if hasattr(model, 'postprocess'):
                tmp = list(map(model.postprocess, tmp))
            out.extend(tmp)
        return out