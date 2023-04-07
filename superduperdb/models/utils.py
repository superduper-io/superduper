import torch

from superduperdb.utils import create_batch, unpack_batch, progressbar, to_device, device_of
from torch.utils import data


class BasicDataset(data.Dataset):
    """
    Basic database iterating over a list of documents and applying a transformation

    :param documents: documents
    :param transform: function
    """
    def __init__(self, documents, transform):
        super().__init__()
        self.documents = documents
        self.transform = transform

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, item):
        return self.transform(self.documents[item])


class Container(torch.nn.Module):
    """
    Class wrapping a ``torch.nn.Module`` adding preprocessing and postprocessing

    :param preprocessor: preprocessing function
    :param forward: forward pass
    :param postprocessor: postprocessing function
    """
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


def create_container(preprocessor=None, forward=None, postprocessor=None):
    if forward is not None:
        assert isinstance(forward, torch.nn.Module)
    if postprocessor is not None:
        assert forward is not None
    if forward is None:
        return TrivialContainer(preprocessor)

    if preprocessor is None:
        preprocessor = lambda x: x
    if postprocessor is None:
        postprocessor = lambda x: x
    return Container(preprocessor=preprocessor, forward=forward, postprocessor=postprocessor)


def apply_model(model, args, single=True, forward='forward', postprocess=True, **kwargs):
    """
    Apply model to args including pre-processing, forward pass and post-processing.

    :param model: model object including methods *preprocess*, *forward* and *postprocess*
    :param args: single or multiple data points over which to evaluate model
    :param single: toggle to apply model to single or multiple (batched) datapoints.
    :param forward: name of the forward pass
    :param postprocess: toggle to ``False`` to get only forward outputs:w
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
        args = to_device(args, device_of(model))
        if hasattr(model, forward):
            singleton_batch = create_batch(args)
            output = getattr(model, forward)(singleton_batch)
            output = to_device(output, 'cpu')
            args = unpack_batch(output)[0]
        if postprocess and hasattr(model, 'postprocess'):
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
            batch = to_device(batch, device_of(model))
            if hasattr(model, forward):
                tmp = getattr(model, forward)(batch)
                tmp = to_device(tmp, 'cpu')
                tmp = unpack_batch(tmp)
            else:
                tmp = unpack_batch(batch)
            if postprocess and hasattr(model, 'postprocess'):
                tmp = list(map(model.postprocess, tmp))
            out.extend(tmp)
        return out