from __future__ import annotations

import concurrent.futures
import dataclasses as dc
import inspect
import multiprocessing
import typing as t
from abc import abstractmethod

import tqdm

from superduperdb import logging
from superduperdb.backends.base.metadata import NonExistentMetadataError
from superduperdb.backends.base.query import CompoundSelect
from superduperdb.backends.ibis.field_types import FieldType
from superduperdb.backends.ibis.query import IbisCompoundSelect, Table
from superduperdb.backends.query_dataset import CachedQueryDataset, QueryDataset
from superduperdb.base.document import Document
from superduperdb.base.serializable import Serializable
from superduperdb.components.component import Component, ensure_initialized
from superduperdb.components.datatype import DataType, dill_lazy
from superduperdb.components.metric import Metric
from superduperdb.components.schema import Schema
from superduperdb.jobs.job import ComponentJob, Job
from superduperdb.misc.annotations import public_api

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.dataset import Dataset


EncoderArg = t.Union[DataType, FieldType, None]
ModelInputType = t.Union[str, t.List[str], t.Tuple[t.List[str], t.Dict[str, str]]]
Signature = t.Literal['*args', '**kwargs', '*args,**kwargs', 'singleton']


def objectmodel(
    item,
    identifier: t.Optional[str] = None,
    datatype=None,
    model_update_kwargs: t.Optional[t.Dict] = None,
    flatten: bool = False,
    output_schema: t.Optional[Schema] = None,
):
    """
    When a class is wrapped with this decorator,
    the instantiated class comes out as an `ObjectModel`.

    :param cls: Class to wrap.
    """
    if callable(item):
        return ObjectModel(
            identifier=identifier or item.__name__,
            object=item,
            datatype=datatype,
            model_update_kwargs=model_update_kwargs or {},
            flatten=flatten,
            output_schema=output_schema,
        )

    else:

        def factory(
            *args,
            identifier: t.Optional[str],
            datatype=None,
            model_update_kwargs: t.Optional[t.Dict] = None,
            flatten: bool = False,
            output_schema: t.Optional[Schema] = None,
            **kwargs,
        ):
            model_update_kwargs = model_update_kwargs or {}
            return ObjectModel(
                identifier=identifier or item.__class__.__name__,
                object=item(*args, **kwargs),
                datatype=datatype,
                model_update_kwargs=model_update_kwargs,
                flatten=flatten,
                output_schema=output_schema,
            )

        return factory


class Inputs:
    def __init__(self, params):
        self.params = params

    def __len__(self):
        return len(self.params)

    def __getattr__(self, attr):
        return self.params[attr]

    def get_kwargs(self, args):
        kwargs = {}
        for k, arg in zip(self.params, args):
            kwargs[k] = arg
        return kwargs


class CallableInputs(Inputs):
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


@dc.dataclass(kw_only=True)
class Trainer(Component):
    """
    Training configuration object, containing all settings necessary for a particular
    learning-task use-case to be serialized and initiated. The object is ``callable``
    and returns a class which may be invoked to apply training.
    """

    type_id: t.ClassVar[str] = 'trainer'

    @abstractmethod
    def fit(
        self,
        model: _Fittable,
        db: Datalayer,
        train_dataset: QueryDataset,
        valid_dataset: QueryDataset,
    ):
        pass


@dc.dataclass(kw_only=True)
class _Validator:
    metrics: t.Sequence[Metric] = ()
    valid_X: t.Optional[ModelInputType] = None
    validation_sets: t.Sequence[Dataset] = ()
    validation_metrics: t.Optional[t.Dict] = None

    def schedule_jobs(self, db, dependencies: t.Sequence[Job] = ()):
        if self.validation_sets and self.validation_metrics:
            return [
                self.validate_in_db_job(
                    db=db,
                    dependencies=dependencies,
                )
            ]
        return []

    def validate(self, X, dataset: Dataset, metrics: t.Sequence[Metric]):
        mapping = Mapping(X, signature='*args')
        predictions = self.predict(dataset.data)
        targets = list(map(mapping, dataset.data))
        results = {}
        for m in metrics:
            results[m.identifier] = m(predictions, targets)
        return results

    def validate_in_db_job(self, db, dependencies: t.Sequence[Job] = ()):
        job = ComponentJob(
            component_identifier=self.identifier,
            method_name='validate_in_db',
            type_id='model',
            kwargs={},
        )
        job(db, dependencies)
        return job

    def validate_in_db(self, db):
        self.metric_values = {}
        for dataset in self.validation_sets:
            db.add(dataset)
            logging.info(f'Validating on {dataset.identifier}...')
            results = self.validate(
                X=self.valid_X,
                dataset=dataset,
                metrics=self.metrics,
            )
            self.metric_values[f'{dataset.identifier}/{dataset.version}'] = results
        db.add(self)


@dc.dataclass(kw_only=True)
class _Fittable:
    trainer: t.Optional[Trainer] = None
    train_X: t.Optional[ModelInputType] = None
    train_select: t.Optional[CompoundSelect] = None
    train_transform: t.Optional[t.Callable] = None
    metric_values: t.Dict = dc.field(default_factory=lambda: {})
    train_signature: Signature = '*args'
    data_prefetch: bool = False
    prefetch_size: int = 1000
    prefetch_factor: int = 100
    in_memory: bool = True

    def schedule_jobs(self, db, dependencies=()):
        jobs = []
        if self.train_X is not None:
            assert isinstance(self.trainer, Trainer)
            assert self.train_select is not None
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
        job = ComponentJob(
            component_identifier=self.identifier,
            method_name='fit_in_db',
            type_id='model',
            kwargs={},
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
        if self.data_prefetch:
            dataset_cls = CachedQueryDataset
            kwargs['prefetch_size'] = self.prefetch_size
        else:
            dataset_cls = QueryDataset

        dataset = dataset_cls(
            select=select,
            fold=fold,
            db=db,
            mapping=Mapping(X, signature=self.train_signature),
            in_memory=self.in_memory,
            transform=(
                self.train_transform if self.train_transform is not None else None
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
        """
        Fit the model on the given data.

        :param db: The datalayer (optional)
        :param select: Select query to get data
        :param trainer: Trainer to use to handle training details
        :param metrics: The metrics to evaluate on (optional)
        :param validation_sets: The validation ``Dataset`` instances to use (optional)
        """
        train_dataset, valid_dataset = self._create_datasets(
            select=self.train_select,
            X=self.train_X,
            db=db,
        )
        return self.fit(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            db=db,
        )

    def append_metrics(self, d: t.Dict[str, float]) -> None:
        if self.metric_values is not None:
            for k, v in d.items():
                self.metric_values.setdefault(k, []).append(v)


class Mapping:
    def __init__(self, mapping: ModelInputType, signature: Signature):
        self.mapping = self._map_args_kwargs(mapping)
        self.signature = signature

    @property
    def id_key(self):
        out = []
        for arg in self.mapping[0]:
            out.append(arg)
        for k, v in self.mapping[1].items():
            if k.startswith('_outputs.'):
                k = k.split('.')[1]
            out.append(f'{k}={v}')
        return ','.join(out)

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
        """
        >>> r = {'a': 1, 'b': 2}
        >>> self.mapping = [('a', 'b'), {}]
        >>> _Predictor._data_from_input_type(docs)
        ([1, 2], {})
        >>> self.mapping = [('a',), {'b': 'X'}]
        >>> _Predictor._data_from_input_type(docs)
        ([1], {'X': 2})
        """
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


@dc.dataclass(kw_only=True)
class Model(Component):
    # base class for components which can predict.
    """:param datatype: DataType instance
    :param output_schema: Output schema (mapping of encoders)
    :param flatten: Flatten the model outputs
    :param collate_fn: Collate function
    :param model_update_kwargs: The kwargs to use for model update
    :param metrics: The metrics to evaluate on
    :param validation_sets: The validation ``Dataset`` instances to use
    :param predict_kwargs: Additional arguments to use at prediction time
    :param compute_kwargs: Kwargs used for compute backend job submit.
                           Example (Ray backend):
                           compute_kwargs = {'resources':{'CustomResource': 1}}
    """

    type_id: t.ClassVar[str] = 'model'
    signature: Signature = '*args,**kwargs'

    datatype: EncoderArg = None
    output_schema: t.Optional[Schema] = None
    flatten: bool = False
    model_update_kwargs: t.Dict = dc.field(default_factory=dict)
    predict_kwargs: t.Dict = dc.field(default_factory=lambda: {})
    compute_kwargs: t.Dict = dc.field(default_factory=lambda: {})

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)
        if not self.identifier:
            raise Exception('_Predictor identifier must be non-empty')

    @property
    def inputs(self) -> Inputs:
        return Inputs(list(inspect.signature(self.predict_one).parameters.keys()))

    @abstractmethod
    def predict_one(self, *args, **kwargs) -> int:
        """
        Execute a single prediction on a datapoint
        given by positional and keyword arguments.

        :param args: arguments handled by model
        :param kwargs: key-word arguments handled by model
        """
        pass

    @abstractmethod
    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        """
        Execute a single prediction on a datapoint
        given by positional and keyword arguments.

        :param args: arguments handled by model
        :param kwargs: key-word arguments handled by model
        """
        pass

    def _prepare_select_for_predict(self, select, db):
        if isinstance(select, dict):
            select = Serializable.decode(select)
        # TODO logic in the wrong place
        if isinstance(select, Table):
            select = select.to_query()
        if isinstance(select, IbisCompoundSelect):
            from superduperdb.backends.sqlalchemy.metadata import SQLAlchemyMetadata

            assert isinstance(db.metadata, SQLAlchemyMetadata)
            try:
                _ = db.metadata.get_query(str(hash(select)))
            except NonExistentMetadataError:
                logging.info(f'Query {select} not found in metadata, adding...')
                db.metadata.add_query(select, self.identifier)
                logging.info('Done')
        return select

    def predict_in_db_job(
        self,
        X: ModelInputType,
        db: Datalayer,
        predict_id: str,
        select: t.Optional[CompoundSelect],
        ids: t.Optional[t.List[str]] = None,
        max_chunk_size: t.Optional[int] = None,
        dependencies: t.Sequence[Job] = (),
        in_memory: bool = True,
        overwrite: bool = False,
    ):
        """
        Execute a single prediction on a datapoint
        given by positional and keyword arguments as a job.

        :param X: combination of input keys to be mapped to the model
        :param db: SuperDuperDB instance
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
            args=[X],
            kwargs={
                'select': select.dict().encode() if select else None,
                'predict_id': predict_id,
                'ids': ids,
                'max_chunk_size': max_chunk_size,
                'in_memory': in_memory,
                'overwrite': overwrite,
            },
            compute_kwargs=self.compute_kwargs,
        )
        job(db, dependencies=dependencies)
        return job

    def _get_ids_from_select(
        self, *, X, select, db, ids, predict_id: str, overwrite: bool = False
    ):
        predict_ids = []
        if not overwrite:
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
        for r in tqdm.tqdm(db.execute(query)):
            predict_ids.append(str(r[id_field]))

        return predict_ids

    def predict_in_db(
        self,
        X: ModelInputType,
        db: Datalayer,
        predict_id: str,
        select: CompoundSelect,
        ids: t.Optional[t.List[str]] = None,
        max_chunk_size: t.Optional[int] = None,
        in_memory: bool = True,
        overwrite: bool = False,
    ) -> t.Any:
        """
        Execute a single prediction on a datapoint
        given by positional and keyword arguments as a job.

        :param X: combination of input keys to be mapped to the model
        :param db: SuperDuperDB instance
        :param select: CompoundSelect query
        :param ids: Iterable of ids
        :param max_chunk_size: Chunks of data
        :param dependencies: List of dependencies (jobs)
        :param in_memory: Load data into memory or not
        :param overwrite: Overwrite all documents or only new documents
        """
        if isinstance(select, dict):
            select = Serializable.decode(select)
        if isinstance(select, Table):
            select = select.to_query()

        select = self._prepare_select_for_predict(select, db)
        if self.identifier not in db.show('model'):
            logging.info(f'Adding model {self.identifier} to db')
            assert isinstance(self, Component)
            db.add(self)
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
        select: CompoundSelect,
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
        select: CompoundSelect,
        ids: t.List[str],
        in_memory: bool = True,
        max_chunk_size: t.Optional[int] = None,
    ):
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
        outputs = self.predict(dataset)
        outputs = self.encode_outputs(outputs)

        logging.info(f'Adding {len(outputs)} model outputs to `db`')

        assert isinstance(
            self.version, int
        ), 'Version has not been set, can\'t save outputs...'
        select.model_update(
            db=db,
            predict_id=predict_id,
            outputs=outputs,
            ids=ids,
            flatten=self.flatten,
            **self.model_update_kwargs,
        )

    def encode_outputs(self, outputs):
        if isinstance(self.datatype, DataType):
            if self.flatten:
                outputs = [
                    [self.datatype(x).encode() for x in output] for output in outputs
                ]
            else:
                outputs = [self.datatype(x).encode() for x in outputs]
        elif isinstance(self.output_schema, Schema):
            outputs = self.encode_with_schema(outputs)

        return outputs

    def encode_with_schema(self, outputs):
        encoded_outputs = []
        for output in outputs:
            if isinstance(output, dict):
                encoded_outputs.append(self.output_schema(output))
            elif self.flatten:
                encoded_output = [self.output_schema(x) for x in output]
                encoded_outputs.append(encoded_output)
        outputs = encoded_outputs if encoded_outputs else outputs
        return outputs

    def __call__(self, outputs: t.Optional[str] = None, **kwargs):
        from superduperdb.components.graph import IndexableNode

        parent_graph = None
        parent_models = {}
        for k, v in kwargs.items():
            if parent_graph is None:
                parent_graph = v.parent_graph
                parent_models.update(v.parent_models)
            elif parent_graph is not None:
                assert v.parent_graph == parent_graph, 'Cannot include 2 parent graphs'
            parent_graph.add_edge(v.model.identifier, self.identifier, key=k)
            parent_models[v.model.identifier] = v
        return IndexableNode(
            model=self,
            parent_graph=parent_graph,
            parent_models=parent_models,
            identifier=outputs,
        )


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


class Node:
    def __init__(self, position):
        self.position = position


@dc.dataclass
class IndexableNode:
    def __init__(self, types):
        self.types = types

    def __getitem__(self, item):
        assert type(item) in self.types
        return Node(item)


@public_api(stability='stable')
@dc.dataclass(kw_only=True)
class ObjectModel(Model, _Validator):
    """Model component which wraps a model to become serializable
    {_predictor_params}
    :param object: Model object, e.g. sklearn model, etc..
    :param num_workers: Number of workers
    """

    type_id: t.ClassVar[str] = 'model'

    __doc__ = __doc__.format(_predictor_params=Model.__doc__)

    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = (
        ('object', dill_lazy),
    )

    object: t.Any
    num_workers: int = 0
    signature: Signature = '*args,**kwargs'

    @property
    def outputs(self):
        return IndexableNode([int])

    @property
    def inputs(self):
        kwargs = self.predict_kwargs if self.predict_kwargs else {}
        return CallableInputs(self.object, kwargs)

    @property
    def training_keys(self) -> t.List:
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

    def append_metrics(self, d: t.Dict[str, float]) -> None:
        if self.metric_values is not None:
            for k, v in d.items():
                self.metric_values.setdefault(k, []).append(v)

    def _wrapper(self, data):
        args, kwargs = self.handle_input_type(data, self.signature)
        return self.object(*args, **kwargs)

    @ensure_initialized
    def predict_one(self, *args, **kwargs):
        return self.object(*args, **kwargs)

    @ensure_initialized
    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
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


# TODO no longer necessary
@public_api(stability='beta')
@dc.dataclass(kw_only=True)
class APIModel(Model):
    '''{component_params}
    {predictor_params}
    :param model: The model to use, e.g. ``'text-embedding-ada-002'``'''

    __doc__ = __doc__.format(
        component_params=Component.__doc__,
        predictor_params=Model.__doc__,
    )

    model: t.Optional[str] = None
    max_batch_size: int = 8

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)
        if self.model is None:
            assert self.identifier is not None
            self.model = self.identifier

    def _multi_predict(
        self, dataset: t.Union[t.List, QueryDataset], *args, **kwargs
    ) -> t.List:
        """
        Base method to batch generate text from a list of prompts using multi-threading.
        Handles exceptions in _generate method.
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_batch_size
        ) as executor:
            results = list(
                executor.map(
                    lambda x: self.predict_one(x, *args, **kwargs),
                    dataset,  # type: ignore[arg-type]
                )
            )

        return results


@public_api(stability='stable')
@dc.dataclass(kw_only=True)
class QueryModel(Model):
    """
    Model which can be used to query data and return those
    results as pre-computed queries.

    :param select: query used to find data (can include `like`)
    """

    preprocess: t.Optional[t.Callable] = None
    postprocess: t.Optional[t.Callable] = None
    select: CompoundSelect

    @property
    def inputs(self) -> Inputs:
        if self.preprocess is not None:
            return CallableInputs(self.preprocess)
        return Inputs([x.value for x in self.select.variables])

    def predict_one(self, X: t.Dict):
        assert self.db is not None, 'db cannot be None'
        if self.preprocess is not None:
            X = self.preprocess(X)
        # TODO: There's something wrong here
        select = self.select.set_variables(db=self.db, **X)
        out = self.db.execute(select)
        if self.postprocess is not None:
            return self.postprocess(out)
        return out

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        return [self.predict_one(dataset[i]) for i in range(len(dataset))]


@public_api(stability='stable')
@dc.dataclass(kw_only=True)
class SequentialModel(Model):
    """
    Sequential model component which wraps a model to become serializable

    {_predictor_params}
    :param predictors: A list of predictors to use
    """

    __doc__ = __doc__.format(
        _predictor_params=Model.__doc__,
    )
    predictors: t.List[Model]

    signature: t.Optional[str] = '*args,**kwargs'

    def __post_init__(self, artifacts):
        self.signature = self.predictors[0].signature
        self.datatype = self.predictors[-1].datatype
        return super().__post_init__(artifacts)

    @property
    def inputs(self) -> Inputs:
        return self.predictors[0].inputs

    def post_create(self, db: Datalayer):
        for p in self.predictors:
            if isinstance(p, str):
                continue
            p.post_create(db)
        self.on_load(db)
        self.signature = self.predictors[0].signature
        self.datatype = self.predictors[-1].datatype

    def on_load(self, db: Datalayer):
        for i, p in enumerate(self.predictors):
            if isinstance(p, str):
                self.predictors[i] = db.load('model', p)

    def predict_one(self, *args, **kwargs):
        return self.predict([(args, kwargs)])[0]

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        for i, p in enumerate(self.predictors):
            assert isinstance(p, Model), f'Expected _Predictor, got {type(p)}'
            if i == 0:
                out = p.predict(dataset)
            else:
                out = p.predict(out)
        return out
