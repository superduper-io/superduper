from __future__ import annotations

import concurrent.futures
import dataclasses as dc
import inspect
import multiprocessing
import os
import re
import typing as t
from abc import abstractmethod
from functools import wraps

import requests
import tqdm

from superduperdb import logging
from superduperdb.backends.base.query import Query
from superduperdb.backends.ibis.field_types import FieldType
from superduperdb.backends.query_dataset import CachedQueryDataset, QueryDataset
from superduperdb.base.document import Document
from superduperdb.base.enums import DBType
from superduperdb.base.exceptions import DatabackendException
from superduperdb.base.leaf import LeafMeta
from superduperdb.components.component import Component, ensure_initialized
from superduperdb.components.datatype import DataType, dill_lazy
from superduperdb.components.metric import Metric
from superduperdb.components.schema import Schema
from superduperdb.jobs.job import ComponentJob, Job

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.dataset import Dataset


EncoderArg = t.Union[DataType, FieldType, None]
ModelInputType = t.Union[str, t.List[str], t.Tuple[t.List[str], t.Dict[str, str]]]
Signature = t.Literal['*args', '**kwargs', '*args,**kwargs', 'singleton']


def model(
    item: t.Optional[t.Callable] = None,
    identifier: t.Optional[str] = None,
    datatype=None,
    model_update_kwargs: t.Optional[t.Dict] = None,
    flatten: bool = False,
    output_schema: t.Optional[Schema] = None,
    num_workers: int = 0,
):
    """Decorator to wrap a function with `ObjectModel`.

    When a function is wrapped with this decorator,
    the function comes out as an `ObjectModel`.

    :param item: Callable to wrap with `ObjectModel`.
    :param identifier: Identifier for the `ObjectModel`.
    :param datatype: Datatype for the model outputs.
    :param model_update_kwargs: Dictionary to define update kwargs.
    :param flatten: If `True`, flatten the outputs and save.
    :param output_schema: Schema for the model outputs.
    :param num_workers: Number of workers to use for parallel processing
    """
    if item is not None and (inspect.isclass(item) or callable(item)):
        if inspect.isclass(item):

            def object_model_factory(*args, **kwargs):
                object_ = item(*args, **kwargs)
                return ObjectModel(
                    object=object_,
                    identifier=identifier or object_.__class__.__name__,
                )

            return object_model_factory
        else:
            assert callable(item)
            return ObjectModel(
                identifier=item.__name__,
                object=item,
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
                        flatten=flatten,
                        output_schema=output_schema,
                        num_workers=num_workers,
                    )

                return object_model_factory
            else:
                assert callable(item)
                return ObjectModel(
                    identifier=identifier or item.__name__,
                    object=item,
                    datatype=datatype,
                    model_update_kwargs=model_update_kwargs or {},
                    flatten=flatten,
                    output_schema=output_schema,
                    num_workers=num_workers,
                )

        return decorated_function


class Inputs:
    """Base class to represent the model args and kwargs.

    :param params: List of parameters of the Model object
    """

    def __init__(self, params):
        self.params = params

    def __len__(self):
        return len(self.params)

    def __getattr__(self, attr):
        return self.params[attr]

    def get_kwargs(self, args):
        """Get keyword arguments from positional arguments.

        :param args: Parameters to be converted
        """
        kwargs = {}
        for k, arg in zip(self.params, args):
            kwargs[k] = arg
        return kwargs

    def __call__(self, *args, **kwargs):
        """Get the model args and kwargs."""
        tmp = self.get_kwargs(args)
        return {**tmp, **kwargs}


class CallableInputs(Inputs):
    """Class represents the model callable args and kwargs.

    :param fn: Callable function
    :param predict_kwargs: (optional) predict_kwargs if provided in Model
                        initiation
    """

    def __init__(self, fn, predict_kwargs: t.Dict = {}):
        sig = inspect.signature(fn)
        full_argspec = inspect.getfullargspec(fn)
        self.kwonly = full_argspec.kwonlyargs
        self.args = full_argspec.args

        sig_keys = list(sig.parameters.keys())
        params = []
        for k in sig_keys:
            if k in predict_kwargs or (k == 'kwargs' and sig.parameters[k].kind == 4):
                continue
            params.append(k)

        self.params = params


class Trainer(Component):
    """Trainer component to train a model.

    Training configuration object, containing all settings necessary for a particular
    learning task use-case to be serialized and initiated. The object is ``callable``
    and returns a class which may be invoked to apply training.

    :param key: Model input type key.
    :param select: Model select query for training.
    :param transform: (optional) transform callable.
    :param metric_values: Dictionary for metric defaults.
    :param signature: Model signature.
    :param data_prefetch: Boolean for prefetching data before forward pass.
    :param prefetch_size: Prefetch batch size.
    :param prefetch_factor: Prefetch factor for data prefetching.
    :param in_memory: If training in memory.
    :param compute_kwargs: Kwargs for compute backend.
    """

    type_id: t.ClassVar[str] = 'trainer'
    key: ModelInputType
    select: Query
    transform: t.Optional[t.Callable] = None
    metric_values: t.Dict = dc.field(default_factory=lambda: {})
    signature: Signature = '*args'
    data_prefetch: bool = False
    prefetch_size: int = 1000
    prefetch_factor: int = 100
    in_memory: bool = True
    compute_kwargs: t.Dict = dc.field(default_factory=dict)

    @abstractmethod
    def fit(
        self,
        model: _Fittable,
        db: Datalayer,
        train_dataset: QueryDataset,
        valid_dataset: QueryDataset,
    ):
        """Fit on the model on training dataset with `valid_dataset` for validation.

        :param model: Model to be fit
        :param db: The datalayer
        :param train_dataset: The training ``Dataset`` instances to use
        :param valid_dataset: The validation ``Dataset`` instances to use
        """
        pass


class Validation(Component):
    """component which represents Validation definition.

    :param metrics: List of metrics for validation
    :param key: Model input type key
    :param datasets: Sequence of dataset.
    """

    type_id: t.ClassVar[str] = 'validation'
    metrics: t.Sequence[Metric] = ()
    key: t.Optional[ModelInputType] = None
    datasets: t.Sequence[Dataset] = ()


@dc.dataclass(kw_only=True)
class _Fittable:
    """:param trainer: Trainer to use to handle training details"""

    trainer: t.Optional[Trainer] = None

    def schedule_jobs(self, db, dependencies=()):
        """Database hook for scheduling jobs.

        :param db: Datalayer instance.
        :param dependencies: List of dependencies.
        """
        jobs = []
        if self.trainer is not None:
            assert isinstance(self.trainer, Trainer)
            assert self.trainer.select is not None
            jobs.append(
                self.fit_in_db_job(
                    db=db,
                    dependencies=dependencies,
                )
            )
        return jobs

    def fit_in_db_job(
        self,
        db: Datalayer,
        dependencies: t.Sequence[Job] = (),
    ):
        """Model fit job in database.

        :param db: Datalayer instance.
        :param dependencies: List of dependent jobs
        """
        if self.trainer:
            compute_kwargs = self.trainer.compute_kwargs or {}
        else:
            compute_kwargs = {}
        job = ComponentJob(
            component_identifier=self.identifier,
            method_name='fit_in_db',
            type_id='model',
            compute_kwargs=compute_kwargs,
        )
        job(db, dependencies)
        return job

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

    def _create_dataset(self, X, db, select, fold=None, **kwargs):
        kwargs = kwargs.copy()
        if self.trainer.data_prefetch:
            dataset_cls = CachedQueryDataset
            kwargs['prefetch_size'] = self.prefetch_size
        else:
            dataset_cls = QueryDataset

        dataset = dataset_cls(
            select=select,
            fold=fold,
            db=db,
            mapping=Mapping(X, signature=self.trainer.signature),
            in_memory=self.trainer.in_memory,
            transform=(
                self.trainer.transform if self.trainer.transform is not None else None
            ),
            **kwargs,
        )
        return dataset

    def fit(
        self,
        train_dataset: QueryDataset,
        valid_dataset: QueryDataset,
        db: Datalayer,
    ):
        """Fit the model on the training dataset with `valid_dataset` for validation.

        :param train_dataset: The training ``Dataset`` instances to use.
        :param valid_dataset: The validation ``Dataset`` instances to use.
        :param db: The datalayer.
        """
        assert isinstance(self.trainer, Trainer)
        if isinstance(self, Component) and self.identifier not in db.show('model'):
            logging.info(f'Adding model {self.identifier} to db')
            assert isinstance(self, Component)
            db.add(self)
        return self.trainer.fit(
            self,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            db=db,
        )

    def fit_in_db(self, db: Datalayer):
        """Fit the model on the given data.

        :param db: The datalayer
        """
        assert isinstance(self.trainer, Trainer)
        train_dataset, valid_dataset = self._create_datasets(
            select=self.trainer.select,
            X=self.trainer.key,
            db=db,
        )
        return self.fit(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            db=db,
        )

    def append_metrics(self, d: t.Dict[str, float]) -> None:
        assert self.trainer is not None
        if self.trainer.metric_values is not None:
            for k, v in d.items():
                self.trainer.metric_values.setdefault(k, []).append(v)


class Mapping:
    """Class to represent model inputs for mapping database collections or tables.

    :param mapping: Mapping that represents a collection or table map.
    :param signature: Signature for the model.
    """

    def __init__(self, mapping: ModelInputType, signature: Signature):
        self.mapping = self._map_args_kwargs(mapping)
        self.signature = signature

    @property
    def id_key(self):
        """Extract the output key for model outputs."""
        outputs = []
        for arg in self.mapping[0]:
            outputs.append(arg)
        for key, value in self.mapping[1].items():
            if key.startswith('_outputs.'):
                key = key.split('.')[1]
            outputs.append(f'{key}={value}')
        return ', '.join(outputs)

    @staticmethod
    def _map_args_kwargs(mapping):
        if isinstance(mapping, str):
            return ([mapping], {})
        elif isinstance(mapping, (list, tuple)) and isinstance(mapping[0], str):
            return (mapping, {})
        elif isinstance(mapping, dict):
            return ((), mapping)
        else:
            assert isinstance(mapping[0], (list, tuple))
            assert isinstance(mapping[1], dict)
            return mapping

    def __call__(self, r):
        """Get the model inputs from the mapping.

        >>> r = {'a': 1, 'b': 2}
        >>> self.mapping = [('a', 'b'), {}]
        >>> _Predictor._data_from_input_type(docs)
        ([1, 2], {})
        >>> self.mapping = [('a',), {'b': 'X'}]
        >>> _Predictor._data_from_input_type(docs)
        ([1], {'X': 2})
        """
        if not isinstance(r, Document):
            r = Document(r)
        args = []
        kwargs = {}
        for key in self.mapping[0]:
            args.append(r[key])
        for k, v in self.mapping[1].items():
            kwargs[v] = r[k]
        args = Document({'_base': args}).unpack()
        kwargs = Document(kwargs).unpack()

        if self.signature == '**kwargs':
            return kwargs
        elif self.signature == '*args':
            return (*args, *list(kwargs.values()))
        elif self.signature == 'singleton':
            if args:
                assert not kwargs
                assert len(args) == 1
                return args[0]
            else:
                assert kwargs
                assert len(kwargs) == 1
                return next(kwargs.values())
        assert self.signature == '*args,**kwargs'
        return args, kwargs


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


class ModelMeta(LeafMeta):
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
        return cls


class Model(Component, metaclass=ModelMeta):
    """Base class for components which can predict.

    :param signature: Model signature.
    :param datatype: DataType instance.
    :param output_schema: Output schema (mapping of encoders).
    :param flatten: Flatten the model outputs.
    :param model_update_kwargs: The kwargs to use for model update.
    :param predict_kwargs: Additional arguments to use at prediction time.
    :param compute_kwargs: Kwargs used for compute backend job submit.
                           Example (Ray backend):
                           compute_kwargs = dict(resources=...).
    :param validation: The validation ``Dataset`` instances to use.
    :param metric_values: The metrics to evaluate on.
    :param num_workers: Number of workers to use for parallel prediction.
    """

    type_id: t.ClassVar[str] = 'model'
    signature: Signature = '*args,**kwargs'
    datatype: EncoderArg = None
    output_schema: t.Optional[Schema] = None
    flatten: bool = False
    model_update_kwargs: t.Dict = dc.field(default_factory=dict)
    predict_kwargs: t.Dict = dc.field(default_factory=dict)
    compute_kwargs: t.Dict = dc.field(default_factory=dict)
    validation: t.Optional[Validation] = None
    metric_values: t.Dict = dc.field(default_factory=dict)
    num_workers: int = 0

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)
        self._is_initialized = False
        if not self.identifier:
            raise Exception('_Predictor identifier must be non-empty')

    @property
    def inputs(self) -> Inputs:
        """Instance of `Inputs` to represent model params."""
        return Inputs(list(inspect.signature(self.predict).parameters.keys()))

    def _wrapper(self, data):
        args, kwargs = self.handle_input_type(data, self.signature)
        return self.predict(*args, **kwargs)

    @abstractmethod
    def predict(self, *args, **kwargs) -> int:
        """Predict on a single data point.

        Execute a single prediction on a data point
        given by positional and keyword arguments.

        :param args: Positional arguments to predict on.
        :param kwargs: Keyword arguments to predict on.
        """
        pass

    def predict_batches(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        """Execute on a series of data points defined in the dataset.

        :param dataset: Series of data points to predict on.
        """
        outputs = []
        if self.num_workers:
            pool = multiprocessing.Pool(processes=self.num_workers)
            for r in pool.map(self._wrapper, dataset):  # type: ignore[arg-type]
                outputs.append(r)
            pool.close()
            pool.join()
        else:
            for i in range(len(dataset)):
                outputs.append(self._wrapper(dataset[i]))
        return outputs

    # TODO handle in job creation
    def _prepare_select_for_predict(self, select, db):
        if isinstance(select, dict):
            select = Document.decode(select).unpack()
        select.set_db(db)
        return select

    def predict_in_db_job(
        self,
        X: ModelInputType,
        db: Datalayer,
        predict_id: str,
        select: t.Optional[Query],
        ids: t.Optional[t.List[str]] = None,
        max_chunk_size: t.Optional[int] = None,
        dependencies: t.Sequence[Job] = (),
        in_memory: bool = True,
        overwrite: bool = False,
    ):
        """Run a prediction job in the database.

        Execute a single prediction on the data points
        given by positional and keyword arguments as a job.

        :param X: combination of input keys to be mapped to the model
        :param db: Datalayer instance
        :param predict_id: Model outputs identifier
        :param select: CompoundSelect query
        :param ids: Iterable of ids
        :param max_chunk_size: Chunks of data
        :param dependencies: List of dependencies (jobs)
        :param in_memory: Load data into memory or not
        :param overwrite: Overwrite all documents or only new documents
        """
        job = ComponentJob(
            component_identifier=self.identifier,
            method_name='predict_in_db',
            type_id='model',
            kwargs={
                'select': select.encode() if select else None,
                'predict_id': predict_id,
                'ids': ids,
                'max_chunk_size': max_chunk_size,
                'in_memory': in_memory,
                'overwrite': overwrite,
                'X': X,
            },
            compute_kwargs=self.compute_kwargs,
        )
        job(db, dependencies=dependencies)
        return job

    def _get_ids_from_select(
        self,
        *,
        X,
        select,
        db: 'Datalayer',
        ids,
        predict_id: str,
        overwrite: bool = False,
    ):
        if not db.databackend.check_output_dest(predict_id):
            query = select.select_ids
        elif not overwrite:
            if ids:
                select = select.select_using_ids(ids)
            if '_outputs' in X:
                X = X.split('.')[1]
            query = select.select_ids_of_missing_outputs(predict_id=predict_id)
        else:
            if ids:
                return ids
            query = select.select_ids
        try:
            id_field = db.databackend.id_field
        except AttributeError:
            id_field = query.table_or_collection.primary_id

        # TODO: Find better solution to support in-memory (pandas)
        # Since pandas has a bug, it cannot join on empty table.
        try:
            id_curr = db.execute(query)
        except DatabackendException:
            id_curr = db.execute(select.select(id_field))

        predict_ids = []
        for r in tqdm.tqdm(id_curr):
            predict_ids.append(str(r[id_field]))
        return predict_ids

    def predict_in_db(
        self,
        X: ModelInputType,
        db: Datalayer,
        predict_id: str,
        select: Query,
        ids: t.Optional[t.List[str]] = None,
        max_chunk_size: t.Optional[int] = None,
        in_memory: bool = True,
        overwrite: bool = False,
    ) -> t.Any:
        """Predict on the data points in the database.

        Execute a single prediction on a data point
        given by positional and keyword arguments as a job.

        :param X: combination of input keys to be mapped to the model
        :param db: Datalayer instance
        :param predict_id: Identifier for saving outputs.
        :param select: CompoundSelect query
        :param ids: Iterable of ids
        :param max_chunk_size: Chunks of data
        :param in_memory: Load data into memory or not
        :param overwrite: Overwrite all documents or only new documents
        """
        message = (
            f'Requesting prediction in db\n'
            f'{self.identifier} with predict_id {predict_id}\n'
            f'Using select {select} and ids {ids}'
        )
        logging.info(message)
        select = self._prepare_select_for_predict(select, db)
        if self.identifier not in db.show('model'):
            logging.info(f'Adding model {self.identifier} to db')
            assert isinstance(self, Component)
            db.apply(self)

        assert isinstance(
            self.version, int
        ), 'Something has gone wrong setting `self.version`'
        predict_ids = self._get_ids_from_select(
            X=X,
            select=select,
            db=db,
            ids=ids,
            overwrite=overwrite,
            predict_id=predict_id,
        )
        return self._predict_with_select_and_ids(
            X=X,
            predict_id=predict_id,
            select=select,
            ids=predict_ids,
            db=db,
            max_chunk_size=max_chunk_size,
            in_memory=in_memory,
        )

    def _prepare_inputs_from_select(
        self,
        X: ModelInputType,
        db: Datalayer,
        select: Query,
        ids,
        in_memory: bool = True,
    ):
        X_data: t.Any
        mapping = Mapping(X, self.signature)
        if in_memory:
            if db is None:
                raise ValueError('db cannot be None')
            docs = list(db.execute(select.select_using_ids(ids)))
            X_data = list(map(lambda x: mapping(x), docs))
        else:
            X_data = QueryDataset(
                select=select,
                ids=ids,
                fold=None,
                db=db,
                in_memory=False,
                mapping=mapping,
            )

        if len(X_data) > len(ids):
            raise Exception(
                'You\'ve specified more documents than unique ids;'
                f' Is it possible that {select.table_or_collection.primary_id}'
                f' isn\'t uniquely identifying?'
            )
        return X_data, mapping

    @staticmethod
    def handle_input_type(data, signature):
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

    def _predict_with_select_and_ids(
        self,
        X: t.Any,
        predict_id: str,
        db: Datalayer,
        select: Query,
        ids: t.List[str],
        in_memory: bool = True,
        max_chunk_size: t.Optional[int] = None,
    ):
        if not ids:
            return

        if max_chunk_size is not None:
            it = 0
            for i in range(0, len(ids), max_chunk_size):
                logging.info(f'Computing chunk {it}/{int(len(ids) / max_chunk_size)}')
                self._predict_with_select_and_ids(
                    X=X,
                    db=db,
                    ids=ids[i : i + max_chunk_size],
                    select=select,
                    max_chunk_size=None,
                    in_memory=in_memory,
                    predict_id=predict_id,
                )
                it += 1
            return

        dataset, mapping = self._prepare_inputs_from_select(
            X=X,
            db=db,
            select=select,
            ids=ids,
            in_memory=in_memory,
        )

        outputs = self.predict_batches(dataset)
        self._infer_auto_schema(outputs, predict_id)
        # TODO implement this so that we can toggle between different ibis/ mongodb
        outputs = self.encode_outputs(outputs)

        logging.info(f'Adding {len(outputs)} model outputs to `db`')

        assert isinstance(
            self.version, int
        ), 'Version has not been set, can\'t save outputs...'

        update = select.model_update(
            db=db,
            predict_id=predict_id,
            outputs=outputs,
            ids=ids,
            flatten=self.flatten,
            **self.model_update_kwargs,
        )
        if update:
            # Don't use auto_schema for inserting model outputs
            if update.type == 'insert':
                update.execute(db=db, auto_schema=False)
            else:
                update.execute(db=db)

    def encode_outputs(self, outputs):
        """Method that encodes outputs of a model for saving in the database.

        :param outputs: outputs to encode.
        """
        # TODO: Fallback when output schema not provided in ibis.

        if isinstance(self.output_schema, Schema):
            return self.encode_with_schema(outputs)
        if isinstance(self.datatype, DataType):
            if self.flatten:
                return [[self.datatype(x) for x in output] for output in outputs]
            else:
                return [self.datatype(x) for x in outputs]
        return outputs

    def _infer_auto_schema(self, outputs, predict_id):
        """
        Infer datatype from outputs of the model.

        :param outputs: Outputs to infer datatype from.
        """
        skip_conds = [
            not self.db.cfg.auto_schema,
            self.datatype is not None,
            self.output_schema is not None,
        ]
        if any(skip_conds):
            return

        output = outputs[0]

        if self.flatten:
            assert isinstance(output, list), 'Flatten is set but output is not list'
            output = output[0]

        # Output schema only for mongodb
        if isinstance(output, dict) and self.db.databackend.db_type == DBType.MONGODB:
            output_schema = self.db.infer_schema(output)
            if output_schema.fields:
                self.output_schema = output_schema
                self.db.apply(self.output_schema)
        else:
            self.datatype = self.db.infer_schema({"data": output}).fields.get(
                "data", None
            )

        if self.datatype is not None and not self.db.databackend.check_output_dest(
            predict_id
        ):
            from superduperdb.components.listener import Listener

            Listener.create_output_dest(self.db, predict_id, self)

        if self.datatype is not None or self.output_schema is not None:
            self.db.replace(self)

    def encode_with_schema(self, outputs):
        """Encode model outputs corresponding to the provided `output_schema`.

        :param outputs: Encode the outputs with the given schema.
        """
        encoded_outputs = []
        for output in outputs:
            if isinstance(output, dict):
                encoded_outputs.append(self.output_schema(output))
            elif self.flatten:
                encoded_output = [self.output_schema(x) for x in output]
                encoded_outputs.append(encoded_output)
        outputs = encoded_outputs if encoded_outputs else outputs
        return outputs

    def __call__(self, *args, outputs: t.Optional[str] = None, **kwargs):
        """Connect the models to build a graph.

        :param args: Arguments to be passed to the model.
        :param outputs: Identifier for the model outputs.
        :param kwargs: Keyword arguments to be passed to the model.
        """
        from superduperdb.components.graph import IndexableNode

        if args:
            predict_params = self.inputs
            assert len(args) <= len(predict_params), 'Too many arguments'
            for i, arg in enumerate(args):
                kwargs[predict_params.params[i]] = arg

        parent_graph = None
        parent_models = {}
        for k, v in kwargs.items():
            if parent_graph is None:
                parent_graph = v.parent_graph
                parent_models.update(v.parent_models)
            elif parent_graph is not None:
                assert (
                    v.parent_graph == parent_graph
                ), 'Cannot include two parent graphs'
            parent_graph.add_edge(v.model.identifier, self.identifier, key=k)
            parent_models[v.model.identifier] = v
        return IndexableNode(
            model=self,
            parent_graph=parent_graph,
            parent_models=parent_models,
            identifier=outputs,
        )

    def to_vector_index(
        self,
        key: ModelInputType,
        select: Query,
        identifier: t.Optional[str] = None,
        predict_kwargs: t.Optional[dict] = None,
        **kwargs,
    ):
        """
        Create a single-model `VectorIndex` from the model.

        :param key: Key to be bound to the model
        :param select: Object for selecting which data is processed
        :param identifier: A string used to identify the model.
        :param predict_kwargs: Keyword arguments to self.model.predict
        :param kwargs: Additional keyword arguments
        """
        identifier = identifier or f'{self.identifier}:vector_index'
        from superduperdb.components.vector_index import VectorIndex

        listener = self.to_listener(
            key=key,
            select=select,
            identifier='',
            predict_kwargs=predict_kwargs,
            **kwargs,
        )
        return VectorIndex(identifier=identifier, indexing_listener=listener)

    def to_listener(
        self,
        key: ModelInputType,
        select: Query,
        identifier='',
        predict_kwargs: t.Optional[dict] = None,
        **kwargs,
    ):
        """Convert the model to a listener.

        :param key: Key to be bound to the model
        :param select: Object for selecting which data is processed
        :param identifier: A string used to identify the model.
        :param predict_kwargs: Keyword arguments to self.model.predict
        :param kwargs: Additional keyword arguments to pass to `Listener`
        """
        from superduperdb.components.listener import Listener

        listener = Listener(
            key=key,
            select=select,
            model=self,
            identifier=identifier,
            predict_kwargs=predict_kwargs or {},
            **kwargs,
        )
        return listener

    def validate(self, key, dataset: Dataset, metrics: t.Sequence[Metric]):
        """Validate `dataset` on metrics.

        :param key: Define input map
        :param dataset: Dataset to run validation on.
        :param metrics: Metrics for performing validation
        """
        mapping1 = Mapping(key[0], self.signature)
        mapping2 = Mapping(key[1], 'singleton')
        inputs = [mapping1(r) for r in dataset.data]
        predictions = self.predict_batches(inputs)
        targets = [mapping2(r) for r in dataset.data]
        results = {}
        for m in metrics:
            results[m.identifier] = m(predictions, targets)
        return results

    def validate_in_db_job(self, db, dependencies: t.Sequence[Job] = ()):
        """Perform a validation job.

        :param db: DataLayer instance
        :param dependencies: dependencies on the job
        """
        job = ComponentJob(
            component_identifier=self.identifier,
            method_name='validate_in_db',
            type_id='model',
            kwargs={},
        )
        job(db, dependencies)
        return job

    def validate_in_db(self, db: Datalayer):
        """Validation job in database.

        :param db: DataLayer instance.
        """
        assert isinstance(self.validation, Validation)
        for dataset in self.validation.datasets:
            logging.info(f'Validating on {dataset.identifier}...')
            db.apply(dataset)
            results = self.validate(
                key=self.validation.key,
                dataset=dataset,
                metrics=self.validation.metrics,
            )
            self.metric_values[f'{dataset.identifier}/{dataset.version}'] = results
        db.replace(self, upsert=True)


@dc.dataclass(kw_only=True)
class _DeviceManaged:
    preferred_devices: t.Sequence[str] = ('cuda', 'mps', 'cpu')
    device: t.Optional[str] = None

    def on_load(self, db: Datalayer) -> None:
        if self.preferred_devices:
            for i, device in enumerate(self.preferred_devices):
                try:
                    self.to(device)
                    self.device = device
                    return
                except Exception:
                    if i == len(self.preferred_devices) - 1:
                        raise
        logging.info(f'Successfully mapped to {self.device}')

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

    """

    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = (
        ('object', dill_lazy),
    )
    object: t.Callable

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
    def outputs(self):
        """Get an instance of ``IndexableNode`` to index outputs."""
        return IndexableNode([int])

    @property
    def inputs(self):
        """A method to get Model callable inputs."""
        kwargs = self.predict_kwargs if self.predict_kwargs else {}
        return CallableInputs(self.object, kwargs)

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
        return self.object(*args, **kwargs)


class APIBaseModel(Model):
    """APIBaseModel component which is used to make the type of API request.

    :param model: The Model to use, e.g. ``'text-embedding-ada-002'``
    :param max_batch_size: Maximum  batch size.
    """

    model: t.Optional[str] = None
    max_batch_size: int = 8

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)
        if self.model is None:
            assert self.identifier is not None
            self.model = self.identifier

    @ensure_initialized
    def _multi_predict(
        self, dataset: t.Union[t.List, QueryDataset], *args, **kwargs
    ) -> t.List:
        """Use multi-threading to predict on a series of data points.

        :param dataset: Series of data points.
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_batch_size
        ) as executor:
            results = list(
                executor.map(
                    lambda x: self.predict(x, *args, **kwargs),
                    dataset,  # type: ignore[arg-type]
                )
            )
        return results


class APIModel(APIBaseModel):
    """APIModel component which is used to make the type of API request.

    :param url: The url to use for the API request
    :param postprocess: Postprocess function to use on the output of the API request
    """

    url: str
    postprocess: t.Optional[t.Callable] = None

    @property
    def inputs(self):
        """Method to get ``Inputs`` instance for model inputs."""
        return Inputs(self.runtime_params)

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)
        self.params['model'] = self.model
        env_variables = re.findall('{([A-Z0-9\_]+)}', self.url)
        runtime_variables = re.findall('{([a-z0-9\_]+)}', self.url)
        runtime_variables = [x for x in runtime_variables if x != 'model']
        self.envs = env_variables
        self.runtime_params = runtime_variables

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
    """

    preprocess: t.Optional[t.Callable] = None
    postprocess: t.Optional[t.Union[t.Callable]] = None
    select: Query
    signature: Signature = '**kwargs'

    @property
    def inputs(self) -> Inputs:
        """Instance of `Inputs` to represent model params."""
        if self.preprocess is not None:
            return CallableInputs(self.preprocess)
        return Inputs([x.value for x in self.select.variables])

    def predict(self, *args, **kwargs):
        """Predict on a single data point.

        Method to perform a single prediction on args and kwargs.
        This method is also used for debugging the model.

        :param args: Positional arguments to predict on.
        :param kwargs: Keyword arguments to predict on.
        """
        assert self.db is not None, 'db cannot be None'
        if self.preprocess is not None:
            kwargs = self.preprocess(**kwargs)
        select = self.select.set_variables(db=self.db, **kwargs)
        out = self.db.execute(select)
        if self.postprocess is not None:
            return self.postprocess(out)
        return out

    def predict_batches(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
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

    models: t.List[Model]

    def __post_init__(self, db, artifacts):
        self.signature = self.models[0].signature
        self.datatype = self.models[-1].datatype
        return super().__post_init__(db, artifacts)

    @property
    def inputs(self) -> Inputs:
        """Instance of `Inputs` to represent model params."""
        return self.models[0].inputs

    def post_create(self, db: Datalayer):
        """Post create hook.

        :param db: Datalayer instance.
        """
        for p in self.models:
            if isinstance(p, str):
                continue
            p.post_create(db)
        self.on_load(db)

    def predict(self, *args, **kwargs):
        """Predict on a single data point.

        Method to do single prediction on args and kwargs.
        This method is also used for debugging the model.

        :param args: Positional arguments to predict on.
        :param kwargs: Keyword arguments to predict on.
        """
        return self.predict_batches([(args, kwargs)])[0]

    def predict_batches(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
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
