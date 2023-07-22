import dataclasses as dc
from functools import cached_property
import io
from contextlib import contextmanager
from typing import Optional, Callable, Union, Dict, List, Any
import typing as t

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from superduperdb.core.artifact import Artifact
from superduperdb.core.metric import Metric
from superduperdb.core.document import Document
from superduperdb.core.encoder import Encodable
from superduperdb.core.model import Model, ModelEnsemble, _TrainingConfiguration
from superduperdb.core.serializable import Serializable
from superduperdb.datalayer.base.datalayer import Datalayer
from superduperdb.datalayer.base.query import Select
from superduperdb.misc.logger import logging
from superduperdb.models.torch.utils import device_of, to_device, eval
from superduperdb.datalayer.query_dataset import QueryDataset


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


@dc.dataclass
class TorchTrainerConfiguration(_TrainingConfiguration):
    objective: t.Optional[t.Union[Artifact, t.Callable]] = None
    loader_kwargs: t.Dict = dc.field(default_factory=dict)
    max_iterations: int = 10**100
    no_improve_then_stop: int = 5
    splitter: t.Optional[Artifact] = None
    download: bool = False
    validation_interval: int = 100
    watch: str = 'objective'
    optimizer_cls: Artifact = Artifact(torch.optim.Adam, serializer='pickle')
    optimizer_kwargs: t.Dict = dc.field(default_factory=dict)
    target_preprocessors: t.Optional[t.Union[Artifact, t.Dict]] = None
    compute_metrics: t.Optional[t.Union[Artifact, t.Callable]] = None

    def __post_init__(self):
        if self.compute_metrics and not isinstance(self.compute_metrics, Artifact):
            self.compute_metrics = Artifact(artifact=self.compute_metrics)

        if self.objective and not isinstance(self.objective, Artifact):
            self.objective = Artifact(artifact=self.objective)

        if self.target_preprocessors and not isinstance(
            self.target_preprocessors, Artifact
        ):
            self.target_preprocessors = Artifact(artifact=self.target_preprocessors)


@dc.dataclass
class Base:
    collate_fn: t.Optional[t.Union[Artifact, t.Callable]] = None
    is_batch: t.Optional[t.Union[Artifact, t.Callable]] = None
    num_directions: int = 2
    metrics: t.Optional[t.List[t.Union[str, Metric]]] = None
    training_select: t.Optional[Select] = None

    @contextmanager
    def evaluating(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def stopping_criterion(self, iteration):
        max_iterations = self.training_configuration.max_iterations
        no_improve_then_stop = self.training_configuration.no_improve_then_stop
        if isinstance(max_iterations, int) and iteration >= max_iterations:
            return True
        if isinstance(no_improve_then_stop, int):
            if self.training_configuration.watch == 'objective':
                to_watch = [-x for x in self.metric_values['objective']]
            else:
                to_watch = self.metric_values[self.training_configuration.watch]

            if max(to_watch[-no_improve_then_stop:]) < max(to_watch):
                logging.info('early stopping triggered!')
                return True
        return False

    def saving_criterion(self):
        if self.training_configuration.watch == 'objective':
            to_watch = [-x for x in self.metric_values['objective']]
        else:
            to_watch = self.metric_values[self.training_configuration.watch]
        if all([to_watch[-1] >= x for x in to_watch[:-1]]):
            return True
        return False

    def _fit(  # type: ignore[override]
        self,
        X: Union[List[str], str],
        y: Optional[Union[List, Any]] = None,
        db: Optional[Datalayer] = None,
        select: Optional[Union[Select, Dict]] = None,
        configuration: Optional[TorchTrainerConfiguration] = None,
        validation_sets: Optional[List[str]] = None,
        metrics: Optional[List[Metric]] = None,
        data_prefetch: bool = False,
    ):
        if configuration is not None:
            self.training_configuration = configuration
        if select is not None:
            if isinstance(select, dict):
                select = Serializable.deserialize(select)
            self.training_select = select  # type: ignore[assignment]
        if validation_sets is not None:
            self.validation_sets = validation_sets
        if metrics is not None:
            self.metrics = metrics

        self.train_X = X
        self.train_y = y

        train_data, valid_data = self._get_data(db=db)
        # ruff: noqa: E501
        loader_kwargs = self.training_configuration.loader_kwargs  # type: ignore[union-attr]
        train_dataloader = DataLoader(train_data, **loader_kwargs)
        valid_dataloader = DataLoader(valid_data, **loader_kwargs)

        return self._fit_with_dataloaders(
            train_dataloader,
            valid_dataloader,
            db=db,  # type: ignore[arg-type]
            validation_sets=validation_sets or [],
        )

    def preprocess(self, r):
        raise NotImplementedError  # implemented in PyTorch wrapper and PyTorch pipeline

    def log(self, **kwargs):
        out = ''
        for k, v in kwargs.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    out += f'{k}/{kk}: {vv}; '
            else:
                out += f'{k}: {v}; '
        logging.info(out)

    def forward(self, X):
        return self.object.artifact(X)

    def extract_batch_key(self, batch, key: Union[List[str], str]):
        if isinstance(key, str):
            return batch[key]
        return [batch[k] for k in key]

    def extract_batch(self, batch):
        if self.train_y is not None:
            return [
                self.extract_batch_key(batch, self.train_X),
                self.extract_batch_key(batch, self.train_y),
            ]
        return [self.extract_batch_key(batch, self.train_X)]

    def take_step(self, batch, optimizers):
        batch = self.extract_batch(batch)
        outputs = self.train_forward(*batch)
        objective_value = self.training_configuration.objective.artifact(*outputs)
        for opt in optimizers:
            opt.zero_grad()
        objective_value.backward()
        for opt in optimizers:
            opt.step()
        return objective_value

    def compute_validation_objective(self, valid_dataloader):
        objective_values = []
        with self.evaluating(), torch.no_grad():
            for batch in valid_dataloader:
                batch = self.extract_batch(batch)
                objective_values.append(
                    self.training_configuration.objective.artifact(
                        *self.train_forward(*batch)
                    ).item()
                )
            return sum(objective_values) / len(objective_values)

    def compute_metrics(self, validation_set, database):
        if validation_set not in self._validation_set_cache:
            self._validation_set_cache[validation_set] = database.load(
                'dataset', validation_set
            )
        data = [r.unpack() for r in self._validation_set_cache[validation_set].data]
        return self.training_configuration.compute_metrics.artifact(
            data,
            metrics=self.metrics,
            model=self,
        )

    def _fit_with_dataloaders(
        self,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        db: Datalayer,  # type: ignore[arg-type]
        validation_sets: List[str],
    ):
        self.train()
        iteration = 0
        while True:
            for batch in train_dataloader:
                train_objective = self.take_step(batch, self.optimizers)
                self.log(fold='TRAIN', iteration=iteration, objective=train_objective)
                # ruff: noqa: E501
                if iteration % self.training_configuration.validation_interval == 0:  # type: ignore[union-attr]
                    valid_loss = self.compute_validation_objective(valid_dataloader)
                    all_metrics = {}
                    for vs in validation_sets:
                        metrics = self.compute_metrics(vs, db)
                        metrics = {f'{vs}/{k}': metrics[k] for k in metrics}
                        all_metrics.update(metrics)
                    all_metrics.update({'objective': valid_loss})
                    self.append_metrics(all_metrics)
                    self.log(fold='VALID', iteration=iteration, **all_metrics)
                    if self.saving_criterion():
                        self.save(db)
                    stop = self.stopping_criterion(iteration)
                    if stop:
                        return
                iteration += 1

    def train_preprocess(self):
        preprocessors = {}
        if isinstance(self.train_X, str):
            preprocessors[self.train_X] = (
                self.preprocess if self.preprocess else lambda x: x
            )
        else:
            for model, X in zip(self.models, self.train_X):
                preprocessors[X] = (
                    model.preprocess if model.preprocess is not None else lambda x: x
                )
        if self.train_y is not None:
            if (
                isinstance(self.train_y, str)
                and self.training_configuration.target_preprocessors
            ):
                preprocessors[
                    self.train_y
                ] = self.training_configuration.target_preprocessors.get(
                    self.train_y, lambda x: x
                )
            elif isinstance(self.train_y, str):
                preprocessors[self.train_y] = lambda x: x
            elif (
                isinstance(self.train_y, list)
                and self.training_configuration.target_preprocessors
            ):
                for y in self.train_y:
                    preprocessors[
                        y
                    ] = self.training_configuration.target_preprocessors.get(
                        y, lambda x: x
                    )

        for k in preprocessors:
            if isinstance(preprocessors[k], Artifact):
                preprocessors[k] = preprocessors[k].artifact
        return lambda r: {k: preprocessors[k](r[k]) for k in preprocessors}

    def _get_data(self, db: Optional[Datalayer]):
        train_data = QueryDataset(
            select=self.training_select,  # type: ignore[arg-type]
            keys=self.training_keys,
            fold='train',
            transform=self.train_preprocess(),
            database=db,
        )
        valid_data = QueryDataset(
            select=self.training_select,  # type: ignore[arg-type]
            keys=self.training_keys,
            fold='valid',
            transform=self.train_preprocess(),
            database=db,
        )
        return train_data, valid_data


@dc.dataclass
class TorchModel(Base, Model):
    preprocess: t.Union[Callable, Artifact, None] = None
    postprocess: t.Union[Callable, Artifact, None] = None
    optimizer_state: t.Optional[Artifact] = None
    forward_method: str = '__call__'
    train_forward_method: str = '__call__'

    def __post_init__(self):
        super().__post_init__()

        self.object.serializer = 'torch'

        if self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state.artifact)
        if self.preprocess and not isinstance(self.preprocess, Artifact):
            self.preprocess = Artifact(artifact=self.preprocess, serializer='dill')
        if self.postprocess and not isinstance(self.postprocess, Artifact):
            self.postprocess = Artifact(artifact=self.postprocess, serializer='dill')

        self._validation_set_cache = {}

    @cached_property
    def optimizer(self):
        return self.training_configuration.optimizer_cls.artifact(
            self.object.artifact.parameters(),
            **self.training_configuration.optimizer_kwargs,
        )

    @property
    def optimizers(self):
        return [self.optimizer]

    def save(self, database: Datalayer):
        self.optimizer_state = Artifact(self.optimizer.state_dict(), serializer='torch')
        database.replace(
            object=self,
            upsert=True,
        )

    @contextmanager
    def evaluating(self):
        yield eval(self)

    def train(self):
        return self.object.artifact.train()

    def eval(self):
        return self.object.artifact.eval()

    def parameters(self):
        return self.object.parameters()

    def state_dict(self):
        return self.object.state_dict()

    @contextmanager
    def saving(self):
        with super().saving():
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
        with torch.no_grad(), eval(self.object.artifact):
            if self.preprocess is not None:
                x = self.preprocess.artifact(x)
            x = to_device(x, device_of(self.object.artifact))
            singleton_batch = create_batch(x)
            method = getattr(self.object.artifact, self.forward_method)
            output = method(singleton_batch)
            output = to_device(output, 'cpu')
            args = unpack_batch(output)[0]
            if hasattr(self.object.artifact, 'postprocess'):
                args = self.object.artifact.postprocess(args)
            return args

    def _predict(self, x, one: bool = False, **kwargs):
        with torch.no_grad(), eval(self.object.artifact):
            if one:
                return self._predict_one(x)
            inputs = BasicDataset(
                x,
                self.preprocess.artifact if self.preprocess else lambda x: x,
            )
            loader = torch.utils.data.DataLoader(inputs, **kwargs)
            out = []
            for batch in tqdm(loader, total=len(loader)):
                batch = to_device(batch, device_of(self.object.artifact))

                method = getattr(self.object.artifact, self.forward_method)
                tmp = method(batch)
                tmp = to_device(tmp, 'cpu')
                tmp = unpack_batch(tmp)
                if self.postprocess is not None:
                    tmp = list(map(self.postprocess.artifact, tmp))
                out.extend(tmp)
            return out

    def train_forward(self, X, y=None):
        method = getattr(self.object.artifact, self.train_forward_method)
        if hasattr(self.object.artifact, 'train_forward'):
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


@dc.dataclass
class TorchModelEnsemble(Base, ModelEnsemble):
    training_configuration: t.Optional[TorchTrainerConfiguration] = None
    optimizer_states: t.Optional[t.List[Artifact]] = None

    def __post_init__(self):
        if self.optimizer_states is not None:
            for i in range(len(self.models)):
                self.optimizers[i].load_state_dict(self.optimizer_states[i].artifact)
        self._validation_set_cache = {}

    @cached_property
    def optimizers(self):
        return [
            self.training_configuration.optimizer_cls.artifact(
                m.object.artifact.parameters(),
                **self.training_configuration.optimizer_kwargs,
            )
            for m in self.models
        ]

    def save(self, database: Datalayer):
        states = []
        for o in self.optimizers:
            states.append(Artifact(o.state_dict(), serializer='torch'))
        self.optimizer_states = states
        database.replace(object=self, upsert=True)

    @contextmanager
    def evaluating(self):
        was_training = self.models[0].object.artifact.training
        try:
            for m in self.models:
                m.eval()
            yield
        finally:
            if was_training:
                for m in self.models:
                    m.train()

    def train(self):
        for m in self.models:
            m.train()

    def train_forward(self, X, y=None):
        out = []
        for i, k in enumerate(self.train_X):
            submodel = self.models[i]
            method = getattr(submodel.object.artifact, submodel.train_forward_method)
            out.append(method(X[i]))
        if y is not None:
            return out, y
        else:
            return out
