from __future__ import annotations

import concurrent.futures
import dataclasses as dc
import inspect
import multiprocessing
import os
import re
import typing as t
from abc import abstractmethod
from collections import defaultdict
from functools import wraps

import requests

from superduper import CFG, logging
from superduper.backends.base.query import Query
from superduper.backends.query_dataset import QueryDataset
from superduper.base.annotations import trigger
from superduper.base.document import Document
from superduper.base.leaf import Leaf
from superduper.components.component import Component, ComponentMeta, ensure_initialized
from superduper.components.metric import Metric
from superduper.components.schema import Schema
from superduper.misc import typing as st

if t.TYPE_CHECKING:
    from superduper.backends.base.cluster import Cluster
    from superduper.base.datalayer import Datalayer
    from superduper.components.dataset import Dataset


# ModelInputType = t.Union[str, t.List[str], t.Tuple[t.List[str], t.Dict[str, str]]]
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
    :param signature: Model signature.
    :param in_memory: If training in memory.
    :param compute_kwargs: Kwargs for compute backend.
    :param validation: Validation object to measure training performance
    """

    _fields = {
        'transform': 'default',
        'select': 'leaf',
        'metric_values': 'json',
        'compute_kwargs': 'json',
        'validation': 'component',
    }

    type_id: t.ClassVar[str] = 'trainer'
    key: st.JSON
    select: st.LeafType
    transform: t.Optional[t.Callable] = None
    metric_values: t.Dict = dc.field(default_factory=lambda: {})
    signature: Signature = '*args'
    in_memory: bool = True
    compute_kwargs: t.Dict = dc.field(default_factory=dict)
    validation: t.Optional[Validation] = None

    @abstractmethod
    def fit(
        self,
        model: Model,
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
    """Component which represents Validation definition.

    :param metrics: List of metrics for validation
    :param key: Model input type key
    :param datasets: Sequence of dataset.
    """

    _fields = {'datasets': 'slist', 'metrics': 'slist'}

    type_id: t.ClassVar[str] = 'validation'
    metrics: t.List[Metric] = dc.field(default_factory=list)
    key: st.JSON
    datasets: t.List[Dataset] = dc.field(default_factory=list)


class Mapping:
    """Class to represent model inputs for mapping database collections or tables.

    :param mapping: Mapping that represents a collection or table map.
    :param signature: Signature for the model.
    """

    def __init__(self, mapping: t.Dict | t.List, signature: Signature):
        self.mapping = self._map_args_kwargs(mapping)
        self.signature = signature

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
        try:
            for k in self.mapping[0]:
                args.append(r[k])
            for k, v in self.mapping[1].items():
                kwargs[v] = r[k]
        except KeyError:
            raise KeyError(f'Key `{k}` not found in document {r}.')

        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, Leaf):
                args[i] = arg.unpack()
        args = tuple(args)

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

    _fields = {
        'datatype': 'leaf',
        'output_schema': 'component',
        'model_update_kwargs': 'json',
        'predict_kwargs': 'json',
        'compute_kwargs': 'json',
        'validation': 'component',
        'metric_values': 'json',
        'trainer': 'component',
    }

    breaks: t.ClassVar[t.Sequence] = ('trainer',)
    type_id: t.ClassVar[str] = 'model'
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
        if not self.identifier:
            raise Exception('_Predictor identifier must be non-empty')
        super().postinit()

    def cleanup(self, db: "Datalayer") -> None:
        """Clean up when the model is deleted.

        :param db: Data layer instance to process.
        """
        super().cleanup(db=db)
        db.cluster.compute.drop(self)

    @property
    def signature(self):
        return self._signature

    @property
    def inputs(self) -> Inputs:
        """Instance of `Inputs` to represent model params."""
        return Inputs(list(inspect.signature(self.predict).parameters.keys()))

    def _wrapper(self, data):
        args, kwargs = self.handle_input_type(data, self.signature)
        return self.predict(*args, **kwargs)

    def declare_component(self, cluster: 'Cluster'):
        """Declare model on compute.

        :param cluster: Cluster instance to declare the model.
        """
        super().declare_component(cluster)
        if self.deploy or self.serve:
            cluster.compute.put(self)

    @abstractmethod
    def predict(self, *args, **kwargs) -> t.Any:
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
        select.db = db
        return select

    # TODO use query chunking not id chunking
    def _get_ids_from_select(
        self,
        *,
        X,
        select,
        ids: t.Sequence[str] | None = None,
        predict_id: str,
        overwrite: bool = False,
    ):
        # TODO why all this complex logic just to get ids
        if not self.db.databackend.check_output_dest(predict_id):
            overwrite = True
        try:
            if not overwrite:
                if ids:
                    select = select.select(select.primary_id)
                # TODO - this is broken
                # query = select.select_ids_of_missing_outputs(predict_id=predict_id)
                predict_ids = select.missing_outputs(predict_id)
            else:
                if ids:
                    return ids
                predict_ids = select.ids()
        except FileNotFoundError:
            # This is case for sql where Table is not created yet
            # and we try to access `db.load('table', name)`.
            return []

        if ids and len(predict_ids) > len(ids):
            raise Exception(
                f'Got {len(predict_ids)} ids from select,'
                f'but {len(ids)} ids were provided'
            )
        return predict_ids

    # TODO - move this logic to `Listenen.run`
    def predict_in_db(
        self,
        X: str | t.List | t.Dict,
        predict_id: str,
        select: Query,
        ids: t.Sequence[str] | None = None,
        max_chunk_size: t.Optional[int] = None,
        in_memory: bool = True,
        overwrite: bool = True,
        flatten: bool = False,
    ) -> t.Any:
        """Predict on the data points in the database.

        Execute a single prediction on a data point
        given by positional and keyword arguments as a job.

        :param X: combination of input keys to be mapped to the model
        :param predict_id: Identifier for saving outputs.
        :param select: CompoundSelect query
        :param ids: Iterable of ids
        :param max_chunk_size: Chunks of data
        :param in_memory: Load data into memory or not
        :param overwrite: Overwrite all documents or only new documents
        :param flatten: Flatten outputs.
        """
        message = (
            f'Requesting prediction in db - '
            f'[{self.identifier}] with predict_id {predict_id}\n'
        )
        logging.info(message)

        message = f'Using select {select} and ids {ids}'
        logging.debug(message)

        select = self._prepare_select_for_predict(select, self.db)

        # TODO ids are not propagated on trigger
        predict_ids = self._get_ids_from_select(
            X=X,
            select=select,
            ids=ids,
            overwrite=overwrite,
            predict_id=predict_id,
        )

        out = self._predict_with_select_and_ids(
            X=X,
            predict_id=predict_id,
            select=select,
            ids=predict_ids,
            max_chunk_size=max_chunk_size,
            in_memory=in_memory,
            flatten=flatten,
        )
        return out

    def _prepare_inputs_from_select(
        self,
        X: str | t.List | t.Dict,
        select: Query,
        ids,
        in_memory: bool = True,
    ):
        X_data: t.Any
        mapping = Mapping(X, self.signature)  # type: ignore[arg-type]

        if in_memory:
            docs = select.subset(ids)
            X_data = list(map(lambda x: mapping(x), docs))
        else:
            assert isinstance(self.db, Datalayer)
            X_data = QueryDataset(
                select=select,
                ids=ids,
                fold=None,
                db=self.db,
                in_memory=False,
                mapping=mapping,
            )

        flat = False
        if 'outputs' in str(select):
            sample = select.get()
            upstream_predict_ids = [
                k for k in sample if k.startswith(CFG.output_prefix)
            ]
            for pid in upstream_predict_ids:
                if self.db.show(uuid=pid.split('__')[-1]).get('flatten', False):
                    flat = True
                    break

        if not flat and len(X_data) > len(ids):
            logging.error("Your select is returning more documents than unique ids.")
            logging.error(f"X_data: {len(X_data)}; ids: {len(ids)}")
            logging.error(f"ids: {ids}")
            logging.error(f"select: {select}")
            raise Exception(
                'You\'ve specified more documents than unique ids;'
                ' Is it possible that the primary_id'
                ' isn\'t uniquely identifying?'
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
        select: Query,
        ids: t.List[str],
        in_memory: bool = True,
        max_chunk_size: t.Optional[int] = None,
        flatten: bool = False,
    ):
        if not ids:
            return []

        if max_chunk_size is not None:
            it = 0
            output_ids = []
            for i in range(0, len(ids), max_chunk_size):
                logging.info(f'Computing chunk {it}/{int(len(ids) / max_chunk_size)}')
                output_ids.extend(
                    self._predict_with_select_and_ids(
                        X=X,
                        ids=ids[i : i + max_chunk_size],
                        select=select,
                        max_chunk_size=None,
                        in_memory=in_memory,
                        predict_id=predict_id,
                        flatten=flatten,
                    )
                )
                it += 1
            return output_ids

        dataset, _ = self._prepare_inputs_from_select(
            X=X,
            select=select,
            ids=ids,
            in_memory=in_memory,
        )

        outputs = self.predict_batches(dataset)
        logging.info(f'Adding {len(outputs)} model outputs to `db`')

        assert isinstance(
            self.version, int
        ), 'Version has not been set, can\'t save outputs...'

        if flatten:
            documents = [
                {
                    self.db.databackend.id_field: self.db.databackend.random_id(),
                    '_source': self.db.databackend.to_id(id),
                    f'{CFG.output_prefix}{predict_id}': sub_output,
                }
                for id, output in zip(ids, outputs)
                for sub_output in output
            ]
        else:
            documents = [
                {
                    self.db.databackend.id_field: self.db.databackend.random_id(),
                    '_source': self.db.databackend.to_id(id),
                    f'{CFG.output_prefix}{predict_id}': output,
                }
                for id, output in zip(ids, outputs)
            ]

        from superduper.base.datalayer import Datalayer

        assert isinstance(self.db, Datalayer)
        return self.db[f'{CFG.output_prefix}{predict_id}'].insert(documents)

    def __call__(self, *args, **kwargs):
        """Connect the models to build a graph.

        :param args: Arguments to be passed to the model.
        :param outputs: Identifier for the model outputs.
        :param kwargs: Keyword arguments to be passed to the model.
        """
        from superduper.misc.eager import SuperDuperData

        is_eager_mode, _ = SuperDuperData.detect_and_get_graph(*args, **kwargs)
        if is_eager_mode:
            return self._eager_call__(*args, **kwargs)

        from superduper.components.graph import IndexableNode

        outputs = kwargs.pop('outputs', None)
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

    def _eager_call__(self, *args, flatten: bool = False, **kwargs):
        from superduper.misc.eager import SuperDuperData, SuperDuperDataType, TrackData

        have_sdd, graph = SuperDuperData.detect_and_get_graph(*args, **kwargs)

        real_args = [
            var.data if isinstance(var, SuperDuperData) else var for var in args
        ]
        real_kwargs = {
            k: var.data if isinstance(var, SuperDuperData) else var
            for k, var in kwargs.items()
        }

        result = self.predict(*real_args, **real_kwargs)
        if not have_sdd:
            return result

        if flatten:
            result = [
                SuperDuperData(
                    var,
                    type=SuperDuperDataType.MODEL_OUTPUT,
                    graph=graph,
                    model=self,
                    flatten=flatten,
                )
                for var in result
            ]
            track_results = result

        else:
            result = SuperDuperData(
                result,
                type=SuperDuperDataType.MODEL_OUTPUT,
                graph=graph,
                model=self,
                flatten=flatten,
            )
            track_results = [result]

        upstream_mapping = defaultdict(list)

        for index, upstream_var in enumerate(args):
            if not isinstance(upstream_var, SuperDuperData):
                if not isinstance(upstream_var, SuperDuperData):
                    # TODO: Support default key-value in listener.key
                    upstream_var = SuperDuperData(
                        upstream_var,
                        type=SuperDuperDataType.CONSTANT,
                        flatten=flatten,
                    )
            upstream_mapping[upstream_var.source].append(TrackData(index, upstream_var))

        for k, upstream_var in kwargs.items():
            if not isinstance(upstream_var, SuperDuperData):
                upstream_var = SuperDuperData(
                    upstream_var,
                    type=SuperDuperDataType.CONSTANT,
                    flatten=flatten,
                )

            upstream_mapping[upstream_var.source].append(TrackData(k, upstream_var))
        assert graph is not None
        for track_result in track_results:
            for upstream_node, track_datas in upstream_mapping.items():
                graph.add_edge(upstream_node, track_result, track_datas)

        return result

    def to_vector_index(
        self,
        key: str | t.List | t.Dict,
        select: Query,
        predict_kwargs: t.Optional[dict] = None,
        identifier: t.Optional[str] = None,
        **kwargs,
    ):
        """
        Create a single-model `VectorIndex` from the model.

        :param key: Key to be bound to the model
        :param select: Object for selecting which data is processed
        :param predict_kwargs: Keyword arguments to self.model.predict
        :param identifier: Identifier for the listener
        :param kwargs: Additional keyword arguments
        """
        from superduper.components.vector_index import VectorIndex

        listener = self.to_listener(
            key=key,
            select=select,
            predict_kwargs=predict_kwargs,
            **kwargs,
        )
        identifier = identifier or f'{self.identifier}/vector_index'
        return VectorIndex(identifier=identifier, indexing_listener=listener)

    def to_listener(
        self,
        key: str | t.List | t.Dict,
        select: Query,
        predict_kwargs: t.Optional[dict] = None,
        identifier: t.Optional[str] = None,
        **kwargs,
    ):
        """Convert the model to a listener.

        :param key: Key to be bound to the model
        :param select: Object for selecting which data is processed
        :param predict_kwargs: Keyword arguments to self.model.predict
        :param identifier: Identifier for the listener
        :param kwargs: Additional keyword arguments to pass to `Listener`
        """
        from superduper.components.listener import Listener

        listener = Listener(
            key=key,
            select=select,
            model=self,
            identifier=identifier or self.identifier,
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
        if isinstance(key, str):
            # metrics are currently functions of 2 inputs.
            key = [key, key]
        mapping1 = Mapping(key[0], self.signature)
        mapping2 = Mapping(key[1], 'singleton')
        inputs = [mapping1(r) for r in dataset.data]
        predictions = self.predict_batches(inputs)
        targets = [mapping2(r) for r in dataset.data]
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

    def _create_dataset(self, X, db, select, fold=None, **kwargs):
        kwargs = kwargs.copy()
        dataset = QueryDataset(
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


# This is if the user would like to
# import the object
class ImportedModel(Model):
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
    object: Leaf
    method: t.Optional[str] = None

    def __post_init__(self, db, example):
        super().__post_init__(db, example)
        self._inferred_signature = None

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
        if self.method is None:
            return self.object(*args, **kwargs)
        return getattr(self.object, self.method)(*args, **kwargs)


class ObjectModel(ImportedModel):
    """A model to wrap a Python object and serialize it."""

    _fields = {'object': 'default'}
    object: t.Callable


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
    def predict_batches(
        self, dataset: t.Union[t.List, QueryDataset], *args, **kwargs
    ) -> t.List:
        """Use multi-threading to predict on a series of data points.

        :param dataset: Series of data points.
        :param args: Positional arguments to predict on.
        :param kwargs: Keyword arguments to predict on.
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
        self.on_load(db)

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


class RAGModel(Model):
    """Model to use for RAG.

    :param prompt_template: Prompt template.
    :param select: Query to retrieve data.
    :param key: Key to use for get text out of documents.
    :param llm: Language model to use.
    """

    breaks: t.ClassVar[t.Sequence] = ('llm', 'prompt_template')

    prompt_template: str
    select: Query
    key: str
    llm: Model

    def _build_prompt(self, query, docs):
        chunks = [doc[self.key] for doc in docs]
        context = "\n\n".join(chunks)
        return self.prompt_template.format(context=context, query=query)

    def predict(self, query: str):
        """Predict on a single query string.

        :param query: Query string.
        """
        from superduper.base.datalayer import Datalayer

        assert isinstance(self.db, Datalayer)
        select = self.select.set_variables(db=self.db, query=query)
        results = [r.unpack() for r in select.execute()]
        prompt = self._build_prompt(query, results)
        return self.llm.predict(prompt)
