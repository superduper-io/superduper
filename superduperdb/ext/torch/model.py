from __future__ import annotations

import dataclasses as dc
import io
import typing as t
from contextlib import contextmanager

from superduperdb.misc.annotations import requires_packages

# TODO: check logic for != version checks
requires_packages(['torch', '2.0.0', None])
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from superduperdb import logging
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.base.datalayer import Datalayer
from superduperdb.components.component import ensure_initialized
from superduperdb.components.dataset import Dataset
from superduperdb.components.datatype import (
    DataType,
    dill_serializer,
)
from superduperdb.components.model import (
    CallableInputs,
    Mapping,
    Signature,
    Trainer,
    _DeviceManaged,
    _Fittable,
    _Predictor,
)
from superduperdb.ext.torch.utils import device_of, eval, to_device


def torchmodel(cls):
    """
    Decorate a `torch.nn.Module` so that when it is invoked,
    the result is a `TorchModel`.

    :param cls: Class to decorate
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
            object=cls(*args, **kwargs),
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
    Basic database iterating over a list of documents and applying a transformation

    :param documents: documents
    :param transform: function
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
            from superduperdb.components.model import Model

            args, kwargs = Model.handle_input_type(out, self.signature)
            return self.transform(*args, **kwargs)
        return out


@dc.dataclass(kw_only=True)
class TorchTrainer(Trainer):
    """
    Configuration for the PyTorch trainer.

    :param objective: Objective function
    :param loader_kwargs: Kwargs for the dataloader
    :param max_iterations: Maximum number of iterations
    :param no_improve_then_stop: Number of iterations to wait for improvement
                                 before stopping
    :param download: Whether to download the data
    :param validation_interval: How often to validate
    :param listen: Which metric to listen to for early stopping
    :param optimizer_cls: Optimizer class
    :param optimizer_kwargs: Kwargs for the optimizer
    :param optimizer_state: Latest state of the optimizer for contined training
    """

    objective: t.Callable
    loader_kwargs: t.Dict = dc.field(default_factory=dict)
    max_iterations: int = 10**100
    no_improve_then_stop: int = 5
    download: bool = False
    validation_interval: int = 100
    listen: str = 'objective'
    optimizer_cls: str = 'Adam'
    optimizer_kwargs: t.Dict = dc.field(default_factory=dict)
    optimizer_state: t.Optional[t.Dict] = None
    collate_fn: t.Optional[t.Callable] = None

    def get_optimizers(self, model):
        cls_ = getattr(torch.optim, self.optimizer_cls)
        optimizer = cls_(model.parameters(), **self.optimizer_kwargs)
        if self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)
            self.optimizer_state = None
        return (optimizer,)

    def _create_loader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            **self.loader_kwargs,
            collate_fn=self.collate_fn,
        )

    def fit(
        self,
        model: TorchModel,
        db: Datalayer,
        train_dataset: QueryDataset,
        valid_dataset: QueryDataset,
    ):
        train_dataloader = self._create_loader(train_dataset)
        valid_dataloader = self._create_loader(valid_dataset)
        return self._fit_with_dataloaders(
            model,
            db,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
        )

    def take_step(self, model, batch, optimizers):
        if model.train_signature == '*args':
            outputs = model.train_forward(*batch)
        elif model.train_signature == 'singleton':
            outputs = model.train_forward(batch)
        elif model.train_signature == '**kwargs':
            outputs = model.train_forward(**batch)
        elif model.train_signature == '*args,**kwargs':
            outputs = model.train_forward(*batch[0], **batch[1])
        objective_value = self.training_configuration.objective(*outputs)
        for opt in optimizers:
            opt.zero_grad()
        objective_value.backward()
        for opt in optimizers:
            opt.step()
        return objective_value

    def _fit_with_dataloaders(
        self,
        model,
        db: Datalayer,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        validation_sets: t.Optional[t.Sequence[t.Union[str, Dataset]]] = None,
    ):
        if validation_sets is None:
            validation_sets = []

        self.model.train()
        iteration = 0

        optimizers = self.get_optimizers(model)

        while True:
            for batch in train_dataloader:
                train_objective = self.take_step(model, batch, optimizers)
                self.log(fold='TRAIN', iteration=iteration, objective=train_objective)

                if iteration % self.validation_interval == 0:
                    valid_loss = self.compute_validation_objective(valid_dataloader)
                    all_metrics = {}
                    for vs in validation_sets:
                        m = model.validate(vs)
                        all_metrics.update(m)
                    all_metrics.update({'objective': valid_loss})
                    self.append_metrics(all_metrics)
                    self.log(fold='VALID', iteration=iteration, **all_metrics)
                    if self.saving_criterion(model):
                        model.changed.append('object')
                        db.replace(model, upsert=True)
                        self.changed.extend(['all_metrics', 'optimizer_state'])
                    stop = self.stopping_criterion(iteration, model)
                    if stop:
                        return
                iteration += 1

    def stopping_criterion(self, iteration, model):
        max_iterations = self.max_iterations
        no_improve_then_stop = self.no_improve_then_stop
        if isinstance(max_iterations, int) and iteration >= max_iterations:
            return True
        if isinstance(no_improve_then_stop, int):
            if self.listen == 'objective':
                to_listen = [-x for x in model.metric_values['objective']]
            else:
                to_listen = model.metric_values[self.listen]

            if max(to_listen[-no_improve_then_stop:]) < max(to_listen):
                logging.info('early stopping triggered!')
                return True
        return False

    def saving_criterion(self, model):
        if self.listen == 'objective':
            to_listen = [-x for x in model.metric_values['objective']]
        else:
            to_listen = model.metric_values[self.listen]
        if all([to_listen[-1] >= x for x in to_listen[:-1]]):
            return True
        return False

    def log(self, **kwargs):
        out = ''
        for k, v in kwargs.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    out += f'{k}/{kk}: {vv}; '
            else:
                out += f'{k}: {v}; '
        logging.info(out)


@dc.dataclass(kw_only=True)
class TorchModel(_Predictor, _Fittable, _DeviceManaged):
    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, DataType]]] = (
        ('object', dill_serializer),
    )

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

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts=artifacts)

        if self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)

        self._validation_set_cache = {}

    @property
    def signature(self):
        if self.preprocess:
            return self.preprocess_signature
        return self.forward_signature

    @signature.setter
    def signature(self, signature):
        if self.preprocess:
            self.preprocess_signature = signature
        else:
            self.forward_signature = signature

    @property
    def inputs(self) -> CallableInputs:
        return CallableInputs(
            self.object.forward if not self.preprocess else self.preprocess, {}
        )

    def to(self, device):
        self.object.to(device)

    def save(self, db: Datalayer):
        with self.saving():
            db.replace(object=self, upsert=True)

    @contextmanager
    def evaluating(self):
        yield eval(self)

    def train(self):
        return self.object.train()

    def eval(self):
        return self.object.eval()

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

    def compute_validation_objective(self, model, valid_dataloader):
        objective_values = []
        with self.evaluating(), torch.no_grad():
            for batch in valid_dataloader:
                objective_values.append(
                    self.objective(*model.train_forward(*batch)).item()
                )
            return sum(objective_values) / len(objective_values)

    @ensure_initialized
    def predict_one(self, *args, **kwargs):
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
    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
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

    def _get_data(self, db: t.Optional[Datalayer]):
        if self.training_select is None:
            raise ValueError('self.training_select cannot be None')
        preprocess = self.preprocess or (lambda x: x)
        train_data = QueryDataset(
            select=self.training_select,
            mapping=Mapping(
                (
                    [self.train_X, self.train_y]  # type: ignore[list-item,arg-type]
                    if self.train_y
                    else self.train_X
                ),
                signature='*args',
            ),
            fold='train',
            transform=(
                preprocess
                if not self.train_y
                else lambda x, y: (preprocess(x), y)  # type: ignore[misc]
            ),
            db=db,
        )
        valid_data = QueryDataset(
            select=self.training_select,
            mapping=Mapping(
                (
                    [self.train_X, self.train_y]  # type: ignore[list-item,arg-type]
                    if self.train_y
                    else self.train_X
                ),
                signature='*args',
            ),
            fold='valid',
            transform=(
                preprocess
                if not self.train_y
                else lambda x, y: (preprocess(x), y)  # type: ignore[misc]
            ),
            db=db,
        )
        return train_data, valid_data


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
    raise TypeError('Only tensors and tuples of tensors recursively supported...')
