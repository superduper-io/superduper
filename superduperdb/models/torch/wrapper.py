import inspect
import io
from contextlib import contextmanager
from typing import Optional, Callable, Union, Dict, List

import torch
from torch.utils import data

from superduperdb.core.base import Placeholder
from superduperdb.core.documents import Document
from superduperdb.core.encoder import Encoder, Encodable
from superduperdb.core.model import Model
from superduperdb.misc import progress
from superduperdb.models.torch.utils import device_of, to_device, eval


class TorchPipeline:
    """
    Sklearn style PyTorch pipeline.

    :param steps: List of ``sklearn`` style steps/ transforms
    :param identifier: Unique identifier
    :param collate_fn: Function for collating batches
    """

    def __init__(
        self,
        identifier,
        steps,
        collate_fn: Optional[Callable] = None,
        is_batch: Optional[Callable] = None,
        encoder: Optional[Union[Encoder, str]] = None,
    ):
        self.steps = steps
        self.identifier = identifier
        self.collate_fn = collate_fn
        self._forward_sequential = None
        self.is_batch = is_batch
        self.type = (
            encoder if isinstance(encoder, Encoder) else Placeholder('type', encoder)
        )

    def __repr__(self):
        lines = [
            'TorchPipeline(steps=[',
            *[f'   {(s[0], s[1])}' for s in self.steps],
            '])',
        ]
        return '\n'.join(lines)

    def _test_if_batch(self, x):
        if self.is_batch is not None:
            return self.is_batch(x)
        if hasattr(self.steps[0][1], '__call__'):
            type = next(
                iter(inspect.signature(self.steps[0][1].__call__).parameters.values())
            ).annotation
        else:
            type = next(
                iter(inspect.signature(self.steps[0][1]).parameters.values())
            ).annotation
        if type != inspect._empty:
            return not isinstance(x, type)
        return isinstance(x, list)

    @property
    def steps(self):
        return self._steps

    def preprocess(self, x):
        for s in self.steps[: self._forward_mark]:
            transform = s[1]
            if hasattr(transform, 'transform'):
                x = transform.transform(x)
            else:
                assert callable(transform)
                x = transform(x)
        return x

    @property
    def preprocess_pipeline(self):
        return TorchPipeline(
            identifier=f'{self.identifier}/preprocess',
            steps=self.steps[: self._forward_mark],
        )

    @property
    def forward_pipeline(self):
        if self._forward_sequential is None:
            forward_steps = self.steps[self._forward_mark : self._post_mark]
            self._forward_sequential = torch.nn.Sequential(
                *[s[1] for s in forward_steps]
            )
        return self._forward_sequential

    def forward(self, x):
        return self._forward_sequential(x)

    @property
    def postprocess_pipeline(self):
        return TorchPipeline(
            identifier=f'{self.identifier}/postprocess',
            steps=self.steps[self._post_mark :],
        )

    def postprocess(self, x):
        for s in self.steps[self._post_mark :]:
            transform = s[1]
            if hasattr(transform, 'transform'):
                x = transform.transform(x)
            else:
                assert callable(transform)
                x = transform(x)
        return x

    def predict(self, x, **kwargs):
        if not self._test_if_batch(x):
            return self._predict_one(x, **kwargs)
        if self.preprocess_pipeline.steps:
            inputs = BasicDataset(x, self.preprocess)
            loader = torch.utils.data.DataLoader(
                inputs, **kwargs, collate_fn=self.collate_fn
            )
        else:
            loader = torch.utils.data.DataLoader(
                x, **kwargs, collate_fn=self.collate_fn
            )
        out = []
        for batch in progress.progressbar(loader, total=len(loader)):
            batch = to_device(batch, device_of(self.forward_pipeline))
            tmp = self.forward(batch)
            tmp = to_device(tmp, 'cpu')
            tmp = unpack_batch(tmp)
            tmp = list(map(self.postprocess, tmp))
            out.extend(tmp)
        return out

    def _predict_one(self, x, **kwargs):
        with torch.no_grad(), eval(self.forward_pipeline):
            x = self.preprocess(x)
            x = to_device(x, device_of(self.forward_pipeline))
            singleton_batch = create_batch(x)
            output = self.forward(singleton_batch)
            output = unpack_batch(output)[0]
            return self.postprocess(output)

    @contextmanager
    def eval(self):
        was_training = self.forward_pipeline.steps[0][1].training
        try:
            for s in self.forward_pipeline.steps:
                s[1].eval()
            yield
        finally:
            if was_training:
                for s in self.forward_pipeline.steps:
                    s[1].eval()

    saving = eval

    @steps.setter
    def steps(self, value):
        self._steps = value
        try:
            self._forward_mark = next(
                i
                for i, s in enumerate(value)
                if isinstance(s[1], torch.nn.Module)
                or isinstance(s[1], torch.jit.ScriptModule)
            )
        except StopIteration:
            self._forward_mark = len(self.steps)

        try:
            self._post_mark = next(
                len(value) - i
                for i, s in enumerate(value[::-1])
                if isinstance(s[1], torch.nn.Module)
                or isinstance(s[1], torch.jit.ScriptModule)
            )
        except StopIteration:
            self._post_mark = len(self.steps)

    def parameters(self):
        for s in self.forward_pipeline.steps:
            yield from s[1].parameters()


class TorchModel(Model):
    def __init__(
        self,
        object,
        identifier,
        encoder=None,
        collate_fn: Optional[Callable] = None,
        num_directions: Optional[Union[Dict, List, int]] = 2,
    ):
        Model.__init__(self, object, identifier, encoder=encoder)
        self.collate_fn = collate_fn
        self.num_directions = num_directions

    def parameters(self):
        return self.object.parameters()

    def state_dict(self):
        return self.object.state_dict()

    @contextmanager
    def saving(self):
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

    def _predict_one(self, x):
        with torch.no_grad(), eval(self.object):
            if hasattr(self.object, 'preprocess'):
                x = self.object.preprocess(x)
            x = to_device(x, device_of(self.object))
            singleton_batch = create_batch(x)
            output = self.object(singleton_batch)
            output = to_device(output, 'cpu')
            args = unpack_batch(output)[0]
            if hasattr(self.object, 'postprocess'):
                args = self.object.postprocess(args)
            return args

    def predict(self, x, **kwargs):
        with torch.no_grad(), eval(self.object):
            if not isinstance(x, list) and not test_if_batch(x, self.num_directions):
                return self._predict_one(x)
            if hasattr(self.object, 'preprocess'):
                inputs = BasicDataset(x, self.object.preprocess)
                loader = torch.utils.data.DataLoader(inputs, **kwargs)
            else:
                loader = torch.utils.data.DataLoader(x, **kwargs)
            out = []
            for batch in progress.progressbar(loader, total=len(loader)):
                batch = to_device(batch, device_of(self.object))
                tmp = self.object(batch)
                tmp = to_device(tmp, 'cpu')
                tmp = unpack_batch(tmp)
                if hasattr(self.object, 'postprocess'):
                    tmp = list(map(self.object.postprocess, tmp))
                out.extend(tmp)
            return out


def test_if_batch(x, num_directions: Union[Dict, int]):
    """
    :param x: item to test whether batch or singleton
    :param num_directions: dictionary to test a leaf node in ``x`` whether batch or not

    >>> test_if_batch(torch.randn(10), 2)
    False
    >>> test_if_batch(torch.randn(2, 10), 2)
    True
    >>> test_if_batch({'x': torch.randn(2, 10)}, {'x': 2})
    True
    >>> test_if_batch({'x': torch.randn(10)}, {'x': 2})
    False
    """
    if isinstance(num_directions, int):
        if len(x.shape) == num_directions:
            return True
        assert len(x.shape) == num_directions - 1
        return False
    else:
        assert len(num_directions.keys()) == 1
        key = next(iter(num_directions.keys()))
        return test_if_batch(x[key], num_directions[key])


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
        document = self.documents[item]
        if isinstance(document, Document):
            document = document.unpack()
        elif isinstance(document, Encodable):
            document = document.x
        return self.transform(document)


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
            return [{k: v[i] for k, v in tmp.items()} for i in range(batch_size)]
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
