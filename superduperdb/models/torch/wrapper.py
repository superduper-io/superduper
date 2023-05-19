import torch
from torch.utils import data

from superduperdb.misc import progress
from superduperdb.models.base import SuperDuperModel
from superduperdb.models.torch.utils import device_of, to_device, eval


class SuperDuperModule(torch.nn.Module, SuperDuperModel):
    def predict_one(self, x, **kwargs):
        with torch.no_grad(), eval(self):
            return apply_model(self, x, single=True, **kwargs)

    def predict(self, x, **kwargs):
        with torch.no_grad(), eval(self):
            return apply_model(self, x, single=False, **kwargs)

    def preprocess(self, x):
        raise NotImplementedError

    def postprocess(self, out):
        raise NotImplementedError


class SuperDuperWrapper(SuperDuperModule):
    def __init__(self, layer, preprocess=None, postprocess=None):
        super().__init__()
        self.layer = layer
        if hasattr(self.layer, 'preprocess') and preprocess is None:
            preprocess = self.layer.preprocess
        if hasattr(self.layer, 'postprocess') and postprocess is None:
            postprocess = self.layer.postprocess
        self._preprocess = preprocess
        self._postprocess = postprocess

    def preprocess(self, x):
        if self._preprocess is None:
            return x
        return self._preprocess(x)

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)

    def postprocess(self, out):
        if self._postprocess is None:
            return out
        return self._postprocess(out)


def apply_model(
    model, args, single=True, forward='forward', postprocess=True, **kwargs
):
    """
    Apply model to args including pre-processing, forward pass
    and post-processing.

    :param model: model object including methods *preprocess*, *forward* and
        *postprocess*
    :param args: single or multiple data points over which to evaluate model
    :param single: toggle to apply model to single or multiple (batched)
        datapoints.
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
    >>> out = apply_model(model, [torch.randn(16), torch.randn(16)],
                          single=False, batch_size=2)
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
    >>> out = apply_model(model, [torch.randn(16), torch.randn(16)],
                          single=False)
    >>> out[0].shape
    torch.Size([32])

    >>> model = SimpleNamespace(
    ...     forward=torch.nn.Linear(16, 4),
    ...     postprocess=lambda x: x.topk(1)[1].item()
    ... )
    >>> out = apply_model(model, [torch.randn(16), torch.randn(16)],
                          single=False, batch_size=2)

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
        for batch in progress.progressbar(loader, total=len(loader)):
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


class BasicDataset(data.Dataset):
    """
    Iterate over a list of documents and apply a transformation

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


def unpack_batch(args):
    """
    Unpack a batch into lines of tensor output.

    :param args: a batch of model outputs

    >>> unpack_batch(torch.randn(1, 10))[0].shape
    torch.Size([10])
    >>> out = unpack_batch([torch.randn(2, 10), torch.randn(2, 3, 5)])
    >>> type(out)
    <class 'list'>
    >>> len(out)
    2
    >>> out = unpack_batch({'a': torch.randn(2, 10), 'b': torch.randn(2, 3, 5)})
    >>> [type(x) for x in out]
    [<class 'dict'>, <class 'dict'>]
    >>> out[0]['a'].shape
    torch.Size([10])
    >>> out[0]['b'].shape
    torch.Size([3, 5])
    >>> out = unpack_batch({'a': {'b': torch.randn(2, 10)}})
    >>> out[0]['a']['b'].shape
    torch.Size([10])
    >>> out[1]['a']['b'].shape
    torch.Size([10])
    """

    if isinstance(args, torch.Tensor):
        return [args[i] for i in range(args.shape[0])]
    else:
        if isinstance(args, list) or isinstance(args, tuple):
            tmp = [unpack_batch(x) for x in args]
            batch_size = len(tmp[0])
            return [[x[i] for x in tmp] for i in range(batch_size)]
        elif isinstance(args, dict):
            tmp = {k: unpack_batch(v) for k, v in args.items()}
            batch_size = len(next(iter(tmp.values())))
            return [
                {k: v[i] for k, v in tmp.items()} for i in range(batch_size)
            ]
        else:  # pragma: no cover
            raise NotImplementedError


def create_batch(args):
    """
    Create a singleton batch in a manner similar to the PyTorch dataloader

    :param args: single data point for batching

    >>> create_batch(3.).shape
    torch.Size([1])
    >>> x, y = create_batch([torch.randn(5), torch.randn(3, 7)])
    >>> x.shape
    torch.Size([1, 5])
    >>> y.shape
    torch.Size([1, 3, 7])
    >>> d = create_batch(({'a': torch.randn(4)}))
    >>> d['a'].shape
    torch.Size([1, 4])
    """
    if isinstance(args, (tuple, list)):
        return tuple([create_batch(x) for x in args])
    if isinstance(args, dict):
        return {k: create_batch(args[k]) for k in args}
    if isinstance(args, torch.Tensor):
        return args.unsqueeze(0)
    if isinstance(args, (float, int)):
        return torch.tensor([args])
    raise TypeError(
        'only tensors and tuples of tensors recursively supported...'
    )  # pragma: no cover
