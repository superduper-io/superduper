from __future__ import annotations

import concurrent.futures
import dataclasses as dc
import inspect
import multiprocessing
import os
import re
import sys
import typing as t
from abc import abstractmethod
from functools import wraps

import requests

from superduper import logging
from superduper.base.annotations import trigger
from superduper.base.query import Query
from superduper.base.schema import Schema
from superduper.components.component import Component, ComponentMeta, ensure_initialized
from superduper.components.metric import Metric
from superduper.misc import typing as st
from superduper.misc.schema import (
    _map_type_to_superduper,
    _safe_resolve_annotation,
    process as process_annotation,
)

if t.TYPE_CHECKING:
    from superduper.backends.base.cluster import Cluster
    from superduper.base.datalayer import Datalayer
    from superduper.components.dataset import Dataset


Signature = t.Literal['*args', '**kwargs', '*args,**kwargs', 'singleton']


def model(
    item: t.Optional[t.Callable] = None,
    identifier: t.Optional[str] = None,
    datatype=None,
    model_update_kwargs: t.Optional[t.Dict] = None,
    output_schema: t.Optional[Schema] = None,
    num_workers: int = 0,
    example: t.Any | None = None,
):
    """Decorator to wrap a function with `ObjectModel`.

    When a function is wrapped with this decorator,
    the function comes out as an `ObjectModel`.

    :param item: Callable to wrap with `ObjectModel`.
    :param identifier: Identifier for the `ObjectModel`.
    :param datatype: Datatype for the model outputs.
    :param model_update_kwargs: Dictionary to define update kwargs.
    :param output_schema: Schema for the model outputs.
    :param num_workers: Number of workers to use for parallel processing
    :param example: Example to auto-determine the schema/ datatype.
    """
    if item is not None and (inspect.isclass(item) or callable(item)):
        if inspect.isclass(item):

            def object_model_factory(*args, **kwargs):
                object_ = item(*args, **kwargs)
                return ObjectModel(
                    object=object_,
                    identifier=identifier or object_.__class__.__name__,
                    datatype=datatype,
                    model_update_kwargs=model_update_kwargs or {},
                    output_schema=output_schema,
                    num_workers=num_workers,
                    example=example,
                )

            return object_model_factory
        else:
            assert callable(item)
            return ObjectModel(
                identifier=item.__name__,
                object=item,
                datatype=datatype,
                model_update_kwargs=model_update_kwargs or {},
                output_schema=output_schema,
                num_workers=num_workers,
                example=example,
            )
    else:

        def decorated_function(item):
            if inspect.isclass(item):

                def object_model_factory(*args, **kwargs):
                    object_ = item(*args, **kwargs)
                    return ObjectModel(
                        identifier=identifier or item.__name__,
                        object=object_,
                        datatype=datatype,
                        model_update_kwargs=model_update_kwargs or {},
                        output_schema=output_schema,
                        num_workers=num_workers,
                        example=example,
                    )

                return object_model_factory
            else:
                assert callable(item)
                return ObjectModel(
                    identifier=identifier or item.__name__,
                    object=item,
                    datatype=datatype,
                    model_update_kwargs=model_update_kwargs or {},
                    output_schema=output_schema,
                    num_workers=num_workers,
                    example=example,
                )

        return decorated_function


# TODO migrate this to its own module
class Trainer(Component):
    """Trainer component to train a model.

    Training configuration object, containing all settings necessary for a particular
    learning task use-case to be serialized and initiated. The object is ``callable``
    and returns a class which may be invoked to apply training.

    :param key: Model input type key.
    :param select: Model select query for training.
    :param transform: (optional) transform callable.
    :param metric_values: Dictionary for metric defaults.
    :param in_memory: If training in memory.
    :param compute_kwargs: Kwargs for compute backend.
    :param validation: Validation object to measure training performance
    """

    key: st.JSON
    select: st.LeafType  # TODO change LeafType to BaseType
    transform: t.Optional[t.Callable] = None
    metric_values: t.Dict = dc.field(default_factory=lambda: {})
    in_memory: bool = True
    compute_kwargs: t.Dict = dc.field(default_factory=dict)
    validation: t.Optional[Validation] = None

    @property
    def signature(self):
        """Signature or the trainer."""
        if isinstance(self.key, str):
            return 'singleton'
        if isinstance(self.key, (list, tuple)) and isinstance(self.key[0], str):
            return '*args'
        if isinstance(self.key, (list, tuple)) and isinstance(
            self.key[0], [list, tuple]
        ):
            assert isinstance(self.key[1], dict)
            return '*args,**kwargs'
        if isinstance(self.key, dict):
            return '**kwargs'
        raise ValueError(f'Could not infer a signature from {self.key}')

    @abstractmethod
    def fit(
        self,
        model: Model,
        db: Datalayer,
        train_dataset: t.List,
        valid_dataset: t.List,
    ):
        """Fit on the model on training dataset with `valid_dataset` for validation.

        :param model: Model to be fit
        :param db: The datalayer
        :param train_dataset: The training ``Dataset`` instances to use
        :param valid_dataset: The validation ``Dataset`` instances to use
        """
        pass


class Validation(Component):
    """Component which represents Validation definition.

    :param metrics: List of metrics for validation
    :param key: Model input type key
    :param datasets: Sequence of dataset.
    """

    metrics: t.List[Metric] = dc.field(default_factory=list)
    key: st.JSON
    datasets: t.List[Dataset] = dc.field(default_factory=list)


def init_decorator(func):
    """Decorator to set _is_initialized to True after init method is called.

    :param func: init function.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self._is_initialized = True
        return result

    return wrapper


class ModelMeta(ComponentMeta):
    """Metaclass for the `Model` class and descendants # noqa."""

    def __new__(mcls, name, bases, dct):
        """Create a new class with merged docstrings # noqa."""
        # Ensure the instance is initialized before calling predict/predict_batches
        if 'predict' in dct:
            dct['predict'] = ensure_initialized(dct['predict'])
        if 'predict_batches' in dct:
            dct['predict_batches'] = ensure_initialized(dct['predict_batches'])
        # If instance call init method, set _is_initialized to True
        if 'init' in dct:
            dct['init'] = init_decorator(dct['init'])
        cls = super().__new__(mcls, name, bases, dct)

        signature = inspect.signature(cls.predict)
        pos = []
        kw = []
        for k in signature.parameters:
            if k == 'self':
                continue
            if signature.parameters[k].default == inspect._empty:
                pos.append(k)
            else:
                kw.append(k)
        if len(pos) == 1 and not kw:
            cls._signature = 'singleton'
        elif pos and not kw:
            cls._signature = '*args'
        elif pos and kw:
            cls._signature = '*args,**kwargs'
        else:
            assert not pos and kw
            cls._signature = '**kwargs'
        return cls


class Model(Component, metaclass=ModelMeta):
    """Base class for components which can predict.

    :param datatype: DataType instance.
    :param output_schema: Output schema (mapping of encoders).
    :param model_update_kwargs: The kwargs to use for model update.
    :param predict_kwargs: Additional arguments to use at prediction time.
    :param compute_kwargs: Kwargs used for compute backend job submit.
                           Example (Ray backend):
                           compute_kwargs = dict(resources=...).
    :param validation: The validation ``Dataset`` instances to use.
    :param metric_values: The metrics to evaluate on.
    :param num_workers: Number of workers to use for parallel prediction.
    :param serve: Creates an http endpoint and serve the model with
                  ``compute_kwargs`` on a distributed cluster.
    :param trainer: `Trainer` instance to use for training.
    :param example: An example to auto-determine the schema/ datatype.
    :param deploy: Creates a standalone class instance on compute cluster.
    """

    breaks: t.ClassVar[t.Sequence] = ('trainer',)

    datatype: str | None = None
    output_schema: t.Optional[t.Dict] = None
    model_update_kwargs: t.Dict = dc.field(default_factory=dict)
    predict_kwargs: t.Dict = dc.field(default_factory=dict)
    compute_kwargs: t.Dict = dc.field(default_factory=dict)
    validation: t.Optional[Validation] = None
    metric_values: t.Dict = dc.field(default_factory=dict)
    num_workers: int = 0
    serve: bool = False
    trainer: t.Optional[Trainer] = None
    example: dc.InitVar[t.Any | None] = None
    deploy: bool = False

    def __post_init__(self, db, example):
        self.example = example
        super().__post_init__(db)

    def postinit(self):
        """Post-initialization method."""
        self._is_initialized = False
        if self.datatype is None:
            annotation = self.predict.__annotations__.get('return')
            if annotation is None:
                self.datatype = 'str'
            else:
                module_globals = sys.modules[__name__].__dict__
                superduper_globals = sys.modules['superduper'].__dict__
                annotation = _safe_resolve_annotation(
                    annotation, {**module_globals, **superduper_globals}
                )
                inferred_annotation, iterable = process_annotation(annotation)
                self.datatype = _map_type_to_superduper(
                    self.__class__.__name__, 'predict', inferred_annotation, iterable
                )

        if not self.identifier:
            raise Exception('_Predictor identifier must be non-empty')
        super().postinit()

    def cleanup(self, db: "Datalayer") -> None:
        """Clean up when the model is deleted.

        :param db: Data layer instance to process.
        """
        super().cleanup(db=db)
        db.cluster.compute.drop(self)

    @staticmethod
    def _infer_signature(object):
        # find positional and key-word parameters from the object
        # using the inspect module
        sig = inspect.signature(object)
        positional = []
        keyword = []
        for k, v in sig.parameters.items():
            if v.default == v.empty:
                positional.append(k)
            else:
                keyword.append(k)
        if not keyword:
            if len(positional) == 1:
                return 'singleton'
            return '*args'
        if not positional:
            return '**kwargs'
        return '*args,**kwargs'

    @property
    def signature(self):
        if self._signature is None:
            self._signature = self._infer_signature(self.predict)
        return self._signature

    def declare_component(self, cluster: 'Cluster'):
        """Declare model on compute.

        :param cluster: Cluster instance to declare the model.
        """
        super().declare_component(cluster)
        # TODO why both of these options??
        if self.deploy or self.serve:
            cluster.scheduler.compute_put_component(self)

    @abstractmethod
    def predict(self, *args, **kwargs) -> t.Any:
        """Predict on a single data point.

        Execute a single prediction on a data point
        given by positional and keyword arguments.

        :param args: Positional arguments to predict on.
        :param kwargs: Keyword arguments to predict on.
        """
        pass

    def _wrapper(self, item):
        """Wrap the item with the model.

        :param item: Item to wrap.
        """
        if self.signature == 'singleton':
            return self.predict(item)
        if self.signature == '*args':
            assert isinstance(item, (list, tuple))
            return self.predict(*item)
        if self.signature == '**kwargs':
            assert isinstance(item, dict)
            return self.predict(**item)
        if self.signature == '*args,**kwargs':
            assert isinstance(item, (list, tuple))
            assert isinstance(item[0], (list, tuple))
            assert isinstance(item[1], dict)
            return self.predict(*item[0], **item[1])
        raise ValueError(f'Unexpected signature {self.signature}')

    def predict_batches(self, dataset: t.List) -> t.List:
        """Execute on a series of data points defined in the dataset.

        :param dataset: Series of data points to predict on.
        """
        outputs = []
        if self.num_workers:
            pool = multiprocessing.Pool(processes=self.num_workers)
            for r in pool.map(self._wrapper, dataset):
                outputs.append(r)
            pool.close()
            pool.join()
        else:
            for i in range(len(dataset)):
                outputs.append(self._wrapper(dataset[i]))
        return outputs

    @staticmethod
    def check_input_type(data, signature):
        """Method to transform data with respect to signature.

        :param data: Data to be transformed
        :param signature: Data signature for transforming
        """
        if signature == 'singleton':
            return (data,), {}
        elif signature == '*args':
            return data, {}
        elif signature == '**kwargs':
            return (), data
        elif signature == '*args,**kwargs':
            return data[0], data[1]
        else:
            raise ValueError(
                f'Unexpected signature {data}: '
                f'Possible values: \'*args\', \'**kwargs\', '
                '\'singleton\', \'*args,**kwargs\'.'
            )

    def validate(self, key, dataset: Dataset, metrics: t.Sequence[Metric]):
        """Validate `dataset` on metrics.

        :param key: Define input map
        :param dataset: Dataset to run validation on.
        :param metrics: Metrics for performing validation
        """
        if isinstance(key, str):
            # metrics are currently functions of 2 inputs.
            key = [key, key]
        inputs = self._map_inputs(self.signature, dataset.data, key[0])
        targets = self._map_inputs('singleton', dataset.data, key[1])
        predictions = self.predict_batches(inputs)
        results = {}
        for m in metrics:
            results[m.identifier] = m(predictions, targets)
        return results

    @trigger('apply', depends='fit_in_db', requires='validation')
    def validate_in_db(self):
        """Validation job in database.

        :param db: DataLayer instance.
        """
        assert isinstance(self.validation, Validation)
        for dataset in self.validation.datasets:
            logging.info(f'Validating on {dataset.identifier}...')
            results = self.validate(
                key=self.validation.key,
                dataset=dataset,
                metrics=self.validation.metrics,
            )
            self.metric_values[f'{dataset.identifier}/{dataset.version}'] = results

        # TODO create self.save()
        self.db.replace(self)

    def _create_datasets(self, X, db, select):
        train_dataset = self._create_dataset(
            X=X,
            db=db,
            select=select,
            fold='train',
        )
        valid_dataset = self._create_dataset(
            X=X,
            db=db,
            select=select,
            fold='valid',
        )
        return train_dataset, valid_dataset

    def _create_dataset(self, X, db, select, fold=None):
        if fold is not None:
            t = db[select.table]
            select = select.filter(t['_fold'] == fold)
        documents = select.execute()
        return self._map_inputs(self.trainer.signature, documents, X)

    @staticmethod
    def _map_inputs(signature, documents, key):
        if signature == 'singleton':
            if key is None:
                return documents
            return [r[key] for r in documents]
        if signature == '*args':
            out = []
            for r in documents:
                out.append([r[x] for x in key])
            return out
        if signature == '**kwargs':
            out = []
            for r in documents:
                out.append({k: r[v] for k, v in key.items()})
            return out
        if signature == '*args,**kwargs':
            out = []
            for r in documents:
                out.append(
                    (
                        [r[x] for x in key[0]],
                        {k: r[v] for k, v in key[1].items()},
                    )
                )
            return out
        raise ValueError(
            f'Invalid signature: {signature}'
            'Allowed signatures are: singleton, *args, **kwargs, *args,**kwargs'
        )

    def fit(
        self,
        train_dataset: t.List[t.Any],
        valid_dataset: t.List[t.Any],
        db: Datalayer,
    ):
        """Fit the model on the training dataset with `valid_dataset` for validation.

        :param train_dataset: The training ``Dataset`` instances to use.
        :param valid_dataset: The validation ``Dataset`` instances to use.
        :param db: The datalayer.
        """
        assert isinstance(self.trainer, Trainer)
        if isinstance(self, Component) and self.identifier not in db.show(
            self.component
        ):
            logging.info(f'Adding model {self.identifier} to db')
            assert isinstance(self, Component)
            db.apply(self)
        return self.trainer.fit(
            self,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            db=db,
        )

    @trigger('apply', requires='trainer')
    def fit_in_db(self):
        """Fit the model on the given data.

        :param db: The datalayer
        """
        assert isinstance(self.trainer, Trainer)
        train_dataset, valid_dataset = self._create_datasets(
            select=self.trainer.select,
            X=self.trainer.key,
            db=self.db,
        )
        if len(train_dataset) == 0:
            logging.warn('No data found for training, skipping training')
            return
        return self.fit(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            db=self.db,
        )

    def append_metrics(self, d: t.Dict[str, float]) -> None:
        """Append metrics to the model.

        :param d: Dictionary of metrics to append.
        """
        assert self.trainer is not None
        if self.trainer.metric_values is not None:
            for k, v in d.items():
                self.trainer.metric_values.setdefault(k, []).append(v)


@dc.dataclass(kw_only=True)
class _DeviceManaged:
    preferred_devices: t.Sequence[str] = ('cuda', 'mps', 'cpu')
    device: t.Optional[str] = None

    @abstractmethod
    def to(self, device):
        pass


class _Node:
    def __init__(self, position):
        self.position = position


@dc.dataclass
class IndexableNode:
    """
    Base indexable node for `ObjectModel`.

    :param types: Sequence of types
    """

    types: t.Sequence[t.Type]

    def __getitem__(self, item):
        assert type(item) in self.types
        return _Node(item)


class ObjectModel(Model):
    """Model component which wraps a Model to become serializable.

    Example:
    -------
    >>> m = ObjectModel('test', lambda x: x + 2)
    >>> m.predict(2)
    4

    :param num_workers: Number of workers to use for parallel processing
    :param object: Model/ computation object
    :param method: Method to call on the object

    """

    breaks: t.ClassVar[t.Sequence] = ('object', 'trainer')
    object: t.Callable
    method: t.Optional[str] = None

    def __post_init__(self, db, example):
        super().__post_init__(db, example)
        self._inferred_signature = None

    @property
    @ensure_initialized
    def signature(self):
        if self._inferred_signature is None:
            self._inferred_signature = self._infer_signature(self.object)
        return self._inferred_signature

    @property
    def outputs(self):
        """Get an instance of ``IndexableNode`` to index outputs."""
        return IndexableNode([int])

    @property
    def training_keys(self) -> t.List:
        """Retrieve training keys."""
        if isinstance(self.train_X, list):
            out = list(self.train_X)
        elif self.train_X is not None:
            out = [self.train_X]
        if self.train_y is not None:
            if isinstance(self.train_y, list):
                out.extend(self.train_y)
            else:
                out.append(self.train_y)
        return out

    def predict(self, *args, **kwargs):
        """Predict on a single data point.

        Method to execute ``Object`` on args and kwargs.
        This method is also used for debugging the Model.

        :param args: Positional arguments of model
        :param kwargs: Keyword arguments of model
        """
        object = self.object
        if self.method is not None:
            return self.object(*args, **kwargs)
            object = getattr(object, self.method)

        return object(*args, **kwargs)


class APIBaseModel(Model):
    """APIBaseModel component which is used to make the type of API request.

    :param model: The Model to use, e.g. ``'text-embedding-ada-002'``
    :param max_batch_size: Maximum  batch size.
    :param postprocess: Postprocess function to use on the output of the API request
    """

    model: t.Optional[str] = None
    max_batch_size: int = 8
    postprocess: t.Optional[t.Callable] = None

    def postinit(self):
        """Post-initialization method."""
        if self.model is None:
            self.model = self.identifier
        super().postinit()

    @ensure_initialized
    def predict_batches(self, dataset: t.List, *args, **kwargs) -> t.List:
        """Use multi-threading to predict on a series of data points.

        :param dataset: Series of data points.
        :param args: Positional arguments to predict on.
        :param kwargs: Keyword arguments to predict on.
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_batch_size
        ) as executor:
            results = list(executor.map(self._wrapper, dataset))
        return results


class APIModel(APIBaseModel):
    """APIModel component which is used to make the type of API request.

    :param url: The url to use for the API request
    :param postprocess: Postprocess function to use on the output of the API request
    """

    url: str

    def postinit(self):
        """Initialize the model data (e.g. weights etc.)."""
        self.params['model'] = self.model
        env_variables = re.findall(r'{([A-Z0-9\_]+)}', self.url)
        runtime_variables = re.findall(r'{([a-z0-9\_]+)}', self.url)
        runtime_variables = [x for x in runtime_variables if x != 'model']
        self.envs = env_variables
        self.runtime_params = runtime_variables
        super().postinit()

    def build_url(self, params):
        """Get url for the ``APIModel``.

        :param params: url params.
        """
        return self.url.format(**params, **{k: os.environ[k] for k in self.envs})

    def predict(self, *args, **kwargs):
        """Predict on a single data point.

        Method to requests to `url` on args and kwargs.
        This method is also used for debugging the model.

        :param args: Positional arguments to predict on.
        :param kwargs: Keyword arguments to predict on.
        """
        runtime_params = self.inputs(*args, **kwargs)
        out = requests.get(self.build_url(params=runtime_params)).json()
        if self.postprocess is not None:
            out = self.postprocess(out)
        return out


class QueryModel(Model):
    """QueryModel component.

    Model which can be used to query data and return those
    precomputed queries as Results.

    :param preprocess: Preprocess callable
    :param postprocess: Postprocess callable
    :param select: query used to find data (can include `like`)
    :param signature: signature to use
    """

    preprocess: t.Optional[t.Callable] = None
    postprocess: t.Optional[t.Callable] = None
    select: Query
    signature: Signature = '**kwargs'

    def predict(self, *args, **kwargs):
        """Predict on a single data point.

        Method to perform a single prediction on args and kwargs.
        This method is also used for debugging the model.

        :param args: Positional arguments to predict on.
        :param kwargs: Keyword arguments to predict on.
        """
        if args:
            raise Exception('QueryModel does not support positional arguments')
        assert self.db is not None, 'db cannot be None'
        if self.preprocess is not None:
            kwargs = self.preprocess(**kwargs)
        select = self.select.set_variables(db=self.db, **kwargs)
        out = self.db.execute(select)
        if self.postprocess is not None:
            return self.postprocess(out)
        return out

    def predict_batches(self, dataset: t.List) -> t.List:
        """Execute on a series of data points defined in the dataset.

        :param dataset: Series of data points to predict on.
        """
        if isinstance(dataset[0], tuple):
            return [
                self.predict(*dataset[i][0], **dataset[i][1])
                for i in range(len(dataset))
            ]
        elif isinstance(dataset[0], dict):
            return [self.predict(**dataset[i]) for i in range(len(dataset))]
        else:
            raise NotImplementedError


class SequentialModel(Model):
    """Sequential model component which wraps a model to become serializable.

    :param models: A list of models to use
    """

    breaks: t.ClassVar[t.Sequence] = ('models',)
    models: t.List[Model]

    def postinit(self):
        """Post-initialization method."""
        self.datatype = self.models[-1].datatype
        return super().postinit()

    @property
    def signature(self):
        return self.models[0].signature

    def on_create(self, db: Datalayer):
        """Post create hook.

        :param db: Datalayer instance.
        """
        for p in self.models:
            if isinstance(p, str):
                continue
            p.on_create(db)

    def predict(self, *args, **kwargs):
        """Predict on a single data point.

        Method to do single prediction on args and kwargs.
        This method is also used for debugging the model.

        :param args: Positional arguments to predict on.
        :param kwargs: Keyword arguments to predict on.
        """
        for i, p in enumerate(self.models):
            assert isinstance(p, Model), f'Expected `Model`, got {type(p)}'
            if i == 0:
                out = p.predict(*args, **kwargs)
            else:
                if p.signature == 'singleton':
                    out = p.predict(out)
                elif p.signature == '*args':
                    out = p.predict(*out)
                elif p.signature == '**kwargs':
                    out = p.predict(**out)
                else:
                    msg = 'Model defines a predict with no free parameters'
                    assert p.signature == '*args,**kwargs', msg
                    out = p.predict(*out[0], **out[1])
        return out

    def predict_batches(self, dataset: t.List) -> t.List:
        """Execute on series of data point defined in dataset.

        :param dataset: Series of data point to predict on.
        """
        for i, p in enumerate(self.models):
            assert isinstance(p, Model), f'Expected `Model`, got {type(p)}'
            if i == 0:
                out = p.predict_batches(dataset)
            else:
                out = p.predict_batches(out)
        return out


class ModelRouter(Model):
    """ModelRouter component which routes the model to the correct model.

    :param models: A dictionary of models to use
    :param model: The model to use
    """

    breaks: t.ClassVar[t.Sequence] = ('models',)
    models: t.Dict[str, Model]
    model: str

    def _pre_create(self, db):
        if not self.datatype:
            self.datatype = self.models[self.model].datatype

    def predict(self, *args, **kwargs) -> t.Any:
        """Predict on a single data point.

        :param args: Positional arguments to predict on.
        :param kwargs: Keyword arguments to predict on.
        """
        logging.info(f'Predicting with model {self.model}')
        return self.models[self.model].predict(*args, **kwargs)

    def fit(self, *args, **kwargs) -> t.Any:
        """Fit the model on the given data.

        :param args: Arguments to fit on.
        :param kwargs: Keyword arguments to fit on.
        """
        logging.info(f'Fitting with model {self.model}')
        return self.models[self.model].fit(*args, **kwargs)

    def predict_batches(self, dataset) -> t.List:
        """Predict on a series of data points defined in the dataset.

        :param dataset: Series of data points to predict on.
        """
        logging.info(f'Predicting with model {self.model}')
        return self.models[self.model].predict_batches(dataset)

    def init(self, db=None):
        """Initialize the model.

        :param db: DataLayer instance.
        """
        if hasattr(self.models[self.model], 'shape'):
            self.shape = getattr(self.models[self.model], 'shape')
        self.example = self.models[self.model].example
        self.signature = self.models[self.model].signature
        self.models[self.model].init()
