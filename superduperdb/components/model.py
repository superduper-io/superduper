from __future__ import annotations

import dataclasses as dc
import inspect
import multiprocessing
import typing as t
from abc import abstractmethod
from functools import wraps

import tqdm
from sklearn.pipeline import Pipeline

from superduperdb import logging
from superduperdb.backends.base.metadata import NonExistentMetadataError
from superduperdb.backends.base.query import CompoundSelect, Select
from superduperdb.backends.ibis.field_types import FieldType
from superduperdb.backends.ibis.query import IbisCompoundSelect, Table
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.base.document import Document
from superduperdb.base.serializable import Serializable
from superduperdb.components.component import Component, ensure_initialized
from superduperdb.components.datatype import DataType, dill_lazy
from superduperdb.components.metric import Metric
from superduperdb.components.schema import Schema
from superduperdb.jobs.job import ComponentJob, Job
from superduperdb.misc.annotations import public_api
from superduperdb.misc.special_dicts import MongoStyleDict

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.dataset import Dataset


EncoderArg = t.Union[DataType, FieldType, None]
ModelInputType = t.Union[str, t.List[str], t.Tuple[t.List[str], t.Dict[str, str]]]


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
        sig_keys = list(sig.parameters.keys())
        params = []
        for k in sig_keys:
            if k in predict_kwargs or (k == 'kwargs' and sig.parameters[k].kind == 4):
                continue
            params.append(k)
        self.params = params


@dc.dataclass(kw_only=True)
class _TrainingConfiguration(Component):
    """
    Training configuration object, containing all settings necessary for a particular
    learning-task use-case to be serialized and initiated. The object is ``callable``
    and returns a class which may be invoked to apply training.

    :param **kwargs: Key-values pairs, the variables which configure training.
    """

    kwargs: t.Optional[t.Dict] = None
    type_id: t.ClassVar[str] = 'training_configuration'

    def get(self, k, default=None):
        try:
            return getattr(self, k)
        except AttributeError:
            return self.kwargs.get(k, default)


@dc.dataclass(kw_only=True)
class _Fittable:
    training_configuration: t.Union[str, _TrainingConfiguration, None] = None
    train_X: t.Optional[ModelInputType] = None
    train_y: t.Optional[str] = None
    train_select: t.Optional[CompoundSelect] = None
    metric_values: t.Dict = dc.field(default_factory=lambda: {})

    def post_create(self, db: Datalayer) -> None:
        if isinstance(self.training_configuration, str):
            self.training_configuration = db.load(
                'training_configuration', self.training_configuration
            )
        # TODO is this necessary - should be handled by `db.add` automatically?

    def schedule_jobs(self, db, dependencies=()):
        jobs = []
        if self.train_X is not None:
            assert (
                isinstance(self.training_configuration, _TrainingConfiguration)
                or self.training_configuration is None
            )
            assert self.train_select is not None
            jobs.append(
                self.fit(
                    X=self.train_X,
                    y=self.train_y,
                    configuration=self.training_configuration,
                    select=self.train_select,
                    db=db,
                    dependencies=dependencies,
                    metrics=self.metrics,
                    validation_sets=self.validation_sets,
                )
            )
        return jobs

    def _validate(
        self,
        db: Datalayer,
        validation_set: t.Union[Dataset, str],
        metrics: t.Sequence[Metric],
    ):
        if isinstance(validation_set, str):
            from superduperdb.components.dataset import Dataset

            validation_set = t.cast(Dataset, db.load('dataset', validation_set))

        mdicts = [MongoStyleDict(r.unpack()) for r in validation_set.data]
        assert self.train_X is not None
        mapping = Mapping(self.train_X, self.signature)
        dataset = list(map(mapping, mdicts))
        prediction = self.predict(dataset)
        assert self.train_y is not None
        target = [d[self.train_y] for d in mdicts]
        assert isinstance(prediction, list)
        assert isinstance(target, list)
        results = {}

        for m in metrics:
            out = m(prediction, target)
            results[f'{validation_set.identifier}/{m.identifier}'] = out
        return results

    def create_fit_job(
        self,
        X: t.Union[str, t.Sequence[str]],
        select: t.Optional[Select] = None,
        y: t.Optional[str] = None,
        **kwargs,
    ):
        return ComponentJob(
            component_identifier=self.identifier,
            method_name='fit',
            type_id='model',
            args=[X],
            kwargs={
                'y': y,
                'select': select.dict().encode() if select else None,
                **kwargs,
            },
        )

    @abstractmethod
    def _fit(
        self,
        X: t.Any,
        y: t.Optional[t.Any] = None,
        configuration: t.Optional[_TrainingConfiguration] = None,
        data_prefetch: bool = False,
        db: t.Optional[Datalayer] = None,
        metrics: t.Optional[t.Sequence[Metric]] = None,
        select: t.Optional[Select] = None,
        validation_sets: t.Optional[t.Sequence[Dataset]] = None,
    ):
        pass

    def fit(
        self,
        X: t.Any,
        y: t.Optional[t.Any] = None,
        configuration: t.Optional[_TrainingConfiguration] = None,
        data_prefetch: bool = False,
        db: t.Optional[Datalayer] = None,
        dependencies: t.Sequence[Job] = (),
        metrics: t.Optional[t.Sequence[Metric]] = None,
        select: t.Optional[Select] = None,
        validation_sets: t.Optional[t.Sequence[Dataset]] = None,
        **kwargs,
    ) -> t.Optional[Pipeline]:
        """
        Fit the model on the given data.

        :param X: The key of the input data to use for training
        :param y: The key of the target data to use for training
        :param configuration: The training configuration (optional)
        :param data_prefetch: Whether to prefetch the data (optional)
        :param db: The datalayer (optional)
        :param dependencies: The dependencies (optional)
        :param metrics: The metrics to evaluate on (optional)
        :param select: The select to use for training (optional)
        :param validation_sets: The validation ``Dataset`` instances to use (optional)
        """
        if isinstance(select, dict):
            # TODO replace with Document.decode(select)
            select = Serializable.from_dict(select)

        if validation_sets:
            from superduperdb.components.dataset import Dataset

            validation_sets = list(validation_sets)
            for i, vs in enumerate(validation_sets):
                if isinstance(vs, Dataset):
                    assert db is not None
                    db.add(vs)
                    validation_sets[i] = vs

        self.training_configuration = configuration or self.training_configuration

        if db is not None:
            db.add(self)

        if db is not None and db.compute.type == 'distributed':
            return self.create_fit_job(
                X,
                select=select,
                y=y,
                **kwargs,
            )(db=db, dependencies=dependencies)
        else:
            return self._fit(
                X,
                y=y,
                configuration=configuration,
                data_prefetch=data_prefetch,
                db=db,
                metrics=metrics,
                select=select,
                validation_sets=validation_sets,
                **kwargs,
            )

    def append_metrics(self, d: t.Dict[str, float]) -> None:
        if self.metric_values is not None:
            for k, v in d.items():
                self.metric_values.setdefault(k, []).append(v)


@wraps(_TrainingConfiguration)
def TrainingConfiguration(identifier: str, **kwargs):
    return _TrainingConfiguration(identifier=identifier, kwargs=kwargs)


@dc.dataclass
class Signature:
    singleton: t.ClassVar[str] = 'singleton'
    args: t.ClassVar[str] = '*args'
    kwargs: t.ClassVar[str] = '**kwargs'
    args_kwargs: t.ClassVar[str] = '*args,**kwargs'


class Mapping:
    def __init__(self, mapping: ModelInputType, signature: str):
        self.mapping = self._map_args_kwargs(mapping)
        self.signature = signature

    @property
    def id_key(self):
        out = []
        for arg in self.mapping[0]:
            out.append(arg)
        for k, v in self.mapping[1]:
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

        if self.signature == Signature.kwargs:
            return kwargs
        elif self.signature == Signature.args:
            return (*args, *list(kwargs.values()))
        elif self.signature == Signature.singleton:
            if args:
                assert not kwargs
                assert len(args) == 1
                return args[0]
            else:
                assert kwargs
                assert len(kwargs) == 1
                return next(kwargs.values())
        return args, kwargs


@dc.dataclass(kw_only=True)
class _Predictor(Component):
    # Mixin class for components which can predict.
    """:param datatype: DataType instance
    :param output_schema: Output schema (mapping of encoders)
    :param flatten: Flatten the model outputs
    :param collate_fn: Collate function
    :param model_update_kwargs: The kwargs to use for model update
    :param metrics: The metrics to evaluate on
    :param validation_sets: The validation ``Dataset`` instances to use
    :param predict_kwargs: Additional arguments to use at prediction time
    """

    type_id: t.ClassVar[str] = 'model'
    signature: t.ClassVar[str] = Signature.args_kwargs

    datatype: EncoderArg = None
    output_schema: t.Optional[Schema] = None
    flatten: bool = False
    model_update_kwargs: t.Dict = dc.field(default_factory=dict)
    metrics: t.Sequence[Metric] = ()
    validation_sets: t.Optional[t.Sequence[Dataset]] = None
    predict_kwargs: t.Dict = dc.field(default_factory=lambda: {})

    def post_create(self, db):
        output_component = db.databackend.create_model_table_or_collection(self)
        if output_component is not None:
            db.add(output_component)

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
                'ids': ids,
                'max_chunk_size': max_chunk_size,
                'in_memory': in_memory,
                'overwrite': overwrite,
            },
        )
        job(db, dependencies=dependencies)
        return job

    def _get_ids_from_select(self, X, select, db, overwrite: bool = False):
        ids = []
        if not overwrite:
            query = select.select_ids_of_missing_outputs(
                key=X,
                model=self.identifier,
                version=t.cast(int, self.version),
            )
        else:
            query = select.select_ids
        try:
            id_field = db.databackend.id_field
        except AttributeError:
            id_field = query.table_or_collection.primary_id
        for r in tqdm.tqdm(db.execute(query)):
            ids.append(str(r[id_field]))
        return ids

    def predict_in_db(
        self,
        X: ModelInputType,
        db: Datalayer,
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

        self._prepare_select_for_predict(select, db)
        if self.identifier not in db.show('model'):
            logging.info(f'Adding model {self.identifier} to db')
            assert isinstance(self, Component)
            db.add(self)
        assert isinstance(
            self.version, int
        ), 'Something has gone wrong setting `self.version`'

        if ids is None:
            ids = self._get_ids_from_select(
                X, select=select, db=db, overwrite=overwrite
            )

        return self._predict_with_select_and_ids(
            X=X,
            select=select,
            ids=ids,
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
            # TODO add signature to Mapping.__call__
            X_data = list(map(lambda x: mapping(x), docs))
        else:
            # TODO above logic missing in case of not a string
            # idea: add the concept of tuple and dictionary strings to `Document`
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
        if signature == Signature.singleton:
            return (data,), {}
        elif signature == Signature.args:
            return data, {}
        elif signature == Signature.kwargs:
            return (), data
        elif signature == Signature.args_kwargs:
            return data[0], data[1]
        else:
            raise ValueError(
                f'Unexpected signature {data}: '
                f'Possible values {Signature.args_kwargs},'
                f'{Signature.kwargs}, '
                f'{Signature.args}, '
                f'{Signature.singleton}.'
            )

    def _predict_with_select_and_ids(
        self,
        X: t.Any,
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
                )
                it += 1
            return

        dataset, mapping = self._prepare_inputs_from_select(
            X=X, db=db, select=select, ids=ids, in_memory=in_memory
        )
        outputs = self.predict(dataset)
        outputs = self.encode_outputs(outputs)

        logging.info(f'Adding {len(outputs)} model outputs to `db`')

        assert isinstance(
            self.version, int
        ), 'Version has not been set, can\'t save outputs...'
        select.model_update(
            db=db,
            model=self.identifier,
            outputs=outputs,
            key=mapping.id_key,
            version=self.version,
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
class ObjectModel(_Predictor):
    """Model component which wraps a model to become serializable
    {_predictor_params}
    :param object: Model object, e.g. sklearn model, etc..
    :param num_workers: Number of workers
    """

    type_id: t.ClassVar[str] = 'model'

    __doc__ = __doc__.format(_predictor_params=_Predictor.__doc__)

    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = (
        ('object', dill_lazy),
    )

    object: t.Any
    num_workers: int = 0
    signature: str = Signature.args_kwargs  # type: ignore[misc]

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

    def validate(
        self, db, validation_set: t.Union[Dataset, str], metrics: t.Sequence[Metric]
    ):
        """
        Validate model on `db` and validation set.

        :param db: `db` SuperDuperDB instance
        :param validation_set: Dataset on which to validate.
        """
        db.add(self)
        out = self._validate(db, validation_set, metrics)
        if self.metric_values is None:
            raise ValueError('self.metric_values cannot be None')
        self.metric_values.update(out)
        db.metadata.update_object(
            type_id='model',
            identifier=self.identifier,
            version=self.version,
            key='dict.metric_values',
            value=self.metric_values,
        )

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


Model = ObjectModel


@public_api(stability='beta')
@dc.dataclass(kw_only=True)
class APIModel(_Predictor):
    '''{component_params}
    {predictor_params}
    :param model: The model to use, e.g. ``'text-embedding-ada-002'``'''

    __doc__ = __doc__.format(
        component_params=Component.__doc__,
        predictor_params=_Predictor.__doc__,
    )

    model: t.Optional[str] = None

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)
        if self.model is None:
            assert self.identifier is not None
            self.model = self.identifier

    def post_create(self, db: Datalayer) -> None:
        # TODO: This not necessary since added as a subcomponent
        if isinstance(self.output_schema, Schema):
            db.add(self.output_schema)
        # TODO add this logic to pre_create,
        # since then the `.add` clause is not necessary
        output_component = db.databackend.create_model_table_or_collection(self)
        if output_component is not None:
            db.add(output_component)


@public_api(stability='stable')
@dc.dataclass(kw_only=True)
class QueryModel(_Predictor):
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
        if self.preprocess is not None:
            X = self.preprocess(X)
        select = self.select.set_variables(db=self.db, **X)
        out = self.db.execute(select)
        if self.postprocess is not None:
            return self.postprocess(out)
        return out

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        return [self.predict_one(dataset[i]) for i in range(len(dataset))]


@public_api(stability='stable')
@dc.dataclass(kw_only=True)
class SequentialModel(_Predictor):
    """
    Sequential model component which wraps a model to become serializable

    {_predictor_params}
    :param predictors: A list of predictors to use
    """

    __doc__ = __doc__.format(
        _predictor_params=_Predictor.__doc__,
    )
    predictors: t.List[_Predictor]

    @property
    def inputs(self) -> Inputs:
        return self.predictors[0].inputs

    def post_create(self, db: Datalayer):
        for p in self.predictors:
            if isinstance(p, str):
                continue
            p.post_create(db)
        self.on_load(db)

    def on_load(self, db: Datalayer):
        for i, p in enumerate(self.predictors):
            if isinstance(p, str):
                self.predictors[i] = db.load('model', p)

    def predict_one(self, *args, **kwargs):
        return self.predict([(args, kwargs)])[0]

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        for i, p in enumerate(self.predictors):
            assert isinstance(p, _Predictor)
            if i == 0:
                out = p.predict(dataset)
            else:
                out = p.predict(out)
        return out
