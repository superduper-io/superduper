from __future__ import annotations

import dataclasses as dc
import io
import typing as t
from contextlib import contextmanager

import torch
from superduper.backends.query_dataset import QueryDataset
from superduper.base.datalayer import Datalayer
from superduper.components.component import ensure_initialized
from superduper.components.model import (
    CallableInputs,
    Model,
    Signature,
    _DeviceManaged,
)
from torch.utils import data
from tqdm import tqdm

from superduper_torch.utils import device_of, eval, to_device


def torchmodel(class_obj):
    """A decorator to convert a `torch.nn.Module` into a `TorchModel`.

    Decorate a `torch.nn.Module` so that when it is invoked,
    the result is a `TorchModel`.

    :param class_obj: Class to decorate
    """

    def factory(
        identifier: str,
        *args,
        preprocess: t.Optional[t.Callable] = None,
        postprocess: t.Optional[t.Callable] = None,
        collate_fn: t.Optional[t.Callable] = None,
        optimizer_state: t.Optional[t.Any] = None,
        forward_method: str = '__call__',
        train_forward_method: str = '__call__',
        loader_kwargs: t.Dict = dc.field(default_factory=lambda: {}),
        preprocess_signature: Signature = 'singleton',
        forward_signature: Signature = 'singleton',
        postprocess_signature: Signature = 'singleton',
        **kwargs,
    ):
        return TorchModel(
            identifier=identifier,
            object=class_obj(*args, **kwargs),
            preprocess=preprocess,
            postprocess=postprocess,
            collate_fn=collate_fn,
            optimizer_state=optimizer_state,
            forward_method=forward_method,
            train_forward_method=train_forward_method,
            loader_kwargs=loader_kwargs,
            preprocess_signature=preprocess_signature,
            forward_signature=forward_signature,
            postprocess_signature=postprocess_signature,
        )

    return factory


class BasicDataset(data.Dataset):
    """
    Basic database iterating over a list of documents and applying a transformation.

    # noqa
    :param items: items, typically documents
    :param transform: function, typically a preprocess function
    :param signature: signature of the transform function
    """

    def __init__(self, items, transform, signature):
        super().__init__()
        self.items = items
        self.transform = transform
        self.signature = signature

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        out = self.items[item]
        if self.transform is not None:
            from superduper.components.model import Model

            args, kwargs = Model.handle_input_type(out, self.signature)
            return self.transform(*args, **kwargs)
        return out


class TorchModel(Model, _DeviceManaged):
    """Torch model. This class is a wrapper around a PyTorch model.

    :param object: Torch model, e.g. `torch.nn.Module`
    :param preprocess: Preprocess function, the function to apply to the input
    :param preprocess_signature: The signature of the preprocess function
    :param postprocess: The postprocess function, the function to apply to the output
    :param postprocess_signature: The signature of the postprocess function
    :param forward_method: The forward method, the method to call on the model
    :param forward_signature: The signature of the forward method
    :param train_forward_method: Train forward method, the method to call on the model
    :param train_forward_signature: The signature of the train forward method
    :param train_preprocess: Train preprocess function,
                                the function to apply to the input
    :param train_preprocess_signature: The signature of the train preprocess function
    :param collate_fn: The collate function for the dataloader
    :param optimizer_state: The optimizer state
    :param loader_kwargs: The kwargs for the dataloader
    :param trainer: `Trainer` object to train the model
    :param preferred_devices: The order of devices to use
    :param device: The device to be used


    Example:
    -------
    >>> import torch
    >>> from superduper_torch.model import TorchModel
    >>>
    >>> model = TorchModel(
    >>>     object=torch.nn.Linear(32, 1),
    >>>     identifier="test",
    >>>     preferred_devices=("cpu",),
    >>>     postprocess=lambda x: int(torch.sigmoid(x).item() > 0.5),
    >>> )
    >>> model.predict(torch.randn(32))

    """

    _fields = {'object': 'default'}

    object: torch.nn.Module
    preprocess: t.Optional[t.Callable] = None
    preprocess_signature: Signature = 'singleton'
    postprocess: t.Optional[t.Callable] = None
    postprocess_signature: Signature = 'singleton'
    forward_method: str = '__call__'
    forward_signature: Signature = 'singleton'
    train_forward_method: str = '__call__'
    train_forward_signature: Signature = 'singleton'
    train_preprocess: t.Optional[t.Callable] = None
    train_preprocess_signature: Signature = 'singleton'
    collate_fn: t.Optional[t.Callable] = None
    optimizer_state: t.Optional[t.Any] = None
    loader_kwargs: t.Dict = dc.field(default_factory=lambda: {})

    def __post_init__(self, db, example):
        super().__post_init__(db, example=example)

        if self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)

        self._validation_set_cache = {}

    @property
    def signature(self):
        """Get the signature of the model."""
        if self.preprocess:
            return self.preprocess_signature
        return self.forward_signature

    @signature.setter
    def signature(self, signature):
        """Set the signature of the model.

        :param signature: Signature
        """
        if self.preprocess:
            self.preprocess_signature = signature
        else:
            self.forward_signature = signature

    @property
    def inputs(self) -> CallableInputs:
        """Get the inputs callable for the model."""
        return CallableInputs(
            self.object.forward if not self.preprocess else self.preprocess, {}
        )

    def to(self, device):
        """Move the model to a device.

        :param device: Device
        """
        self.object.to(device)

    def save(self, db: Datalayer):
        """Save the model to the database.

        :param db: Datalayer
        """
        with self.saving():
            db.replace(object=self, upsert=True)

    @contextmanager
    def evaluating(self):
        """Context manager for evaluating the model.

        This context manager ensures that the model is in evaluation mode
        """
        yield eval(self)

    def train(self):
        """Set the model to training mode."""
        return self.object.train()

    def eval(self):
        """Set the model to evaluation mode."""
        return self.object.eval()

    def parameters(self):
        """Get the model parameters."""
        return self.object.parameters()

    def state_dict(self):
        """Get the model state dict."""
        return self.object.state_dict()

    @contextmanager
    def saving(self):
        """Context manager for saving the model.

        This context manager ensures that the model is in evaluation mode
        """
        was_training = self.object.training
        try:
            self.object.eval()
            yield
        finally:
            if was_training:
                self.object.train()

    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(self.object, torch.jit.ScriptModule) or isinstance(
            self.object, torch.jit.ScriptFunction
        ):
            f = io.BytesIO()
            torch.jit.save(self.object, f)
            state['_object_bytes'] = f.getvalue()
        return state

    def __setstate__(self, state):
        keys = state.keys()
        for k in keys:
            if k != '_object_bytes':
                self.__dict__[k] = state[k]
            else:
                state.__dict__['object'] = torch.jit.load(
                    io.BytesIO(state.pop('object_bytes'))
                )

    @ensure_initialized
    def predict(self, *args, **kwargs):
        """Predict on a single input.

        :param args: Input arguments
        :param kwargs: Input keyword arguments
        """
        with torch.no_grad(), eval(self.object):
            if self.preprocess is not None:
                out = self.preprocess(*args, **kwargs)
                args, kwargs = self.handle_input_type(out, self.signature)

            args, kwargs = to_device((args, kwargs), self.device)
            args, kwargs = create_batch((args, kwargs))

            method = getattr(self.object, self.forward_method)
            output = method(*args, **kwargs)
            output = to_device(output, 'cpu')
            args = unpack_batch(output)[0]
            if self.postprocess is not None:
                args = self.postprocess(args)
            return args

    @ensure_initialized
    def predict_batches(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        """Predict on a dataset.

        :param dataset: Dataset
        """
        with torch.no_grad(), eval(self.object):
            inputs = BasicDataset(
                items=dataset,
                transform=self.preprocess,
                signature=self.preprocess_signature,
            )
            loader = torch.utils.data.DataLoader(
                inputs, **self.loader_kwargs, collate_fn=self.collate_fn
            )
            out = []
            for batch in tqdm(loader, total=len(loader)):
                batch = to_device(batch, device_of(self.object))
                args, kwargs = self.handle_input_type(batch, self.signature)
                method = getattr(self.object, self.forward_method)
                tmp = method(*args, **kwargs, **self.predict_kwargs)
                tmp = to_device(tmp, 'cpu')
                tmp = unpack_batch(tmp)
                if self.postprocess:
                    tmp = [
                        self.handle_input_type(x, self.postprocess_signature)
                        for x in tmp
                    ]
                    tmp = [self.postprocess(*x[0], **x[1]) for x in tmp]
                out.extend(tmp)
            return out

    def train_forward(self, X, y=None):
        """The forward method for training.

        :param X: Input
        :param y: Target
        """
        X = X.to(self.device)
        if y is not None:
            y = y.to(self.device)

        method = getattr(self.object, self.train_forward_method)
        if hasattr(self.object, 'train_forward'):
            if y is None:
                return method(X)
            else:
                return method(X, y=y)
        else:
            if y is None:
                return (method(X),)
            else:
                return [method(X), y]


def unpack_batch(args):
    """Unpack a batch into lines of tensor output.

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

    if isinstance(args, list) or isinstance(args, tuple):
        tmp = [unpack_batch(x) for x in args]
        batch_size = len(tmp[0])
        return [[x[i] for x in tmp] for i in range(batch_size)]

    if isinstance(args, dict):
        tmp = {k: unpack_batch(v) for k, v in args.items()}
        batch_size = len(next(iter(tmp.values())))
        return [{k: v[i] for k, v in tmp.items()} for i in range(batch_size)]

    raise NotImplementedError


def create_batch(args):
    """Create a singleton batch in a manner similar to the PyTorch dataloader.

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
    raise TypeError('Only tensors and tuples of tensors recursively supported...')
