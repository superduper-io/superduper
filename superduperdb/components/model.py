from __future__ import annotations

import dataclasses as dc
import inspect
import multiprocessing
import typing as t
from abc import abstractmethod
from functools import wraps

import tqdm
from overrides import override
from sklearn.pipeline import Pipeline

from superduperdb import Document, logging
from superduperdb.backends.base.metadata import NonExistentMetadataError
from superduperdb.backends.base.query import CompoundSelect, Select, TableOrCollection
from superduperdb.backends.ibis.field_types import FieldType
from superduperdb.backends.ibis.query import IbisCompoundSelect, Table
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.base.serializable import Serializable
from superduperdb.components.component import Component
from superduperdb.components.datatype import DataType, dill_serializer
from superduperdb.components.metric import Metric
from superduperdb.components.schema import Schema
from superduperdb.jobs.job import ComponentJob, Job
from superduperdb.misc.annotations import public_api
from superduperdb.misc.special_dicts import MongoStyleDict

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.dataset import Dataset

EncoderArg = t.Union[DataType, FieldType, str, None]
XType = t.Union[t.Any, t.List, t.Dict]


class _to_call:
    def __init__(self, callable, **kwargs):
        self.callable = callable
        self.kwargs = kwargs

    def __call__(self, X):
        return self.callable(X, **self.kwargs)


class Inputs:
    def __init__(self, fn, predict_kwargs: t.Dict = {}):
        sig = inspect.signature(fn)
        sig_keys = list(sig.parameters.keys())
        params = []
        for k in sig_keys:
            if k in predict_kwargs or (k == 'kwargs' and sig.parameters[k].kind == 4):
                continue
            params.append(k)

        self.params = {p: p for p in params}

    def __len__(self):
        return len(self.params)

    def __getattr__(self, attr):
        return self.params[attr]

    def get_kwargs(self, args):
        kwargs = {}
        for k, arg in zip(self.params, args):
            kwargs[k] = arg
        return kwargs


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
class _Predictor:
    # Mixin class for components which can predict.
    """:param encoder: Encoder instance
    :param output_schema: Output schema (mapping of encoders)
    :param flatten: Flatten the model outputs
    :param preprocess: Preprocess function
    :param postprocess: Postprocess function
    :param collate_fn: Collate function
    :param batch_predict: Whether to batch predict
    :param takes_context: Whether the model takes context into account
    :param metrics: The metrics to evaluate on
    :param model_update_kwargs: The kwargs to use for model update
    :param validation_sets: The validation ``Dataset`` instances to use
    :param predict_X: The key of the input data to use for .predict
    :param predict_select: The select to use for .predict
    :param predict_max_chunk_size: The max chunk size to use for .predict
    :param predict_kwargs: The kwargs to use for .predict"""

    type_id: t.ClassVar[str] = 'model'

    datatype: EncoderArg = None
    output_schema: t.Optional[Schema] = None
    flatten: bool = False
    preprocess: t.Optional[t.Callable] = None
    postprocess: t.Optional[t.Callable] = None
    collate_fn: t.Optional[t.Callable] = None
    batch_predict: bool = False
    takes_context: bool = False
    metrics: t.Sequence[t.Union[str, Metric, None]] = ()
    model_update_kwargs: t.Dict = dc.field(default_factory=dict)
    validation_sets: t.Optional[t.Sequence[t.Union[str, Dataset]]] = None

    predict_X: t.Optional[str] = None
    predict_select: t.Optional[CompoundSelect] = None
    predict_max_chunk_size: t.Optional[int] = None
    predict_kwargs: t.Optional[t.Dict] = None

    @abstractmethod
    def to_call(self, X, *args, **kwargs):
        """
        The method to use to call prediction. Should be implemented
        by the child class.
        """

    @property
    def inputs(self):
        kwargs = self.predict_kwargs if self.predict_kwargs else {}
        return Inputs(self.object, kwargs)

    def create_predict_job(
        self,
        X: XType,
        select: t.Optional[Select] = None,
        ids: t.Optional[t.Sequence[str]] = None,
        max_chunk_size: t.Optional[int] = None,
        **kwargs,
    ):
        return ComponentJob(
            component_identifier=self.identifier,
            method_name='predict',
            type_id='model',
            args=[X],
            kwargs={
                'select': select.dict().encode() if select else None,
                'ids': ids,
                'max_chunk_size': max_chunk_size,
                **kwargs,
            },
        )

    async def _apredict_one(self, X: t.Any, **kwargs):
        raise NotImplementedError

    async def _apredict(self, X: t.Any, one: bool = False, **kwargs):
        raise NotImplementedError

    def _predict_one(self, X: t.Any, **kwargs) -> int:
        if self.preprocess:
            X = self.preprocess(X)
        output = self.to_call(X, **kwargs)
        if self.postprocess:
            output = self.postprocess(output)
        return output

    def _forward(
        self, X: t.Sequence[int], num_workers: int = 0, **kwargs
    ) -> t.Sequence[int]:
        if self.batch_predict:
            return self.to_call(X, **kwargs)

        outputs = []
        if num_workers:
            to_call = _to_call(self.to_call, **kwargs)
            pool = multiprocessing.Pool(processes=num_workers)
            for r in pool.map(to_call, X):
                outputs.append(r)
            pool.close()
            pool.join()
        else:
            for r in X:
                outputs.append(self.to_call(r, **kwargs))
        return outputs

    def _predict(self, X: t.Any, one: bool = False, **predict_kwargs):
        if one:
            return self._predict_one(X)

        if self.preprocess:
            X = [self.preprocess(i) for i in X]
        elif self.preprocess is not None:
            raise ValueError('Bad preprocess')
        if self.collate_fn:
            X = self.collate_fn(X)

        outputs = self._forward(X, **predict_kwargs)

        if self.postprocess:
            outputs = [self.postprocess(o) for o in outputs]
        elif self.postprocess is not None:
            raise ValueError('Bad postprocess')

        return outputs

    def predict(
        self,
        X: XType,
        db: t.Optional[Datalayer] = None,
        select: t.Optional[CompoundSelect] = None,
        ids: t.Optional[t.List[str]] = None,
        max_chunk_size: t.Optional[int] = None,
        dependencies: t.Sequence[Job] = (),
        listen: bool = False,
        one: bool = False,
        context: t.Optional[t.Dict] = None,
        insert_to: t.Optional[t.Union[TableOrCollection, str]] = None,
        key: t.Optional[t.Union[t.Dict, t.List, str]] = None,
        in_memory: bool = True,
        overwrite: bool = False,
        **kwargs,
    ) -> t.Any:
        was_added = self.db is not None
        db = self.db or db

        if one:
            assert select is None, 'select must be None when ``one=True`` (direct call)'

        if isinstance(select, dict):
            select = Serializable.decode(select)

        if isinstance(select, Table):
            select = select.to_query()

        if db is not None:
            if isinstance(select, IbisCompoundSelect):
                from superduperdb.backends.sqlalchemy.metadata import SQLAlchemyMetadata

                assert isinstance(db.metadata, SQLAlchemyMetadata)
                try:
                    _ = db.metadata.get_query(str(hash(select)))
                except NonExistentMetadataError:
                    logging.info(f'Query {select} not found in metadata, adding...')
                    db.metadata.add_query(select, self.identifier)
                    logging.info('Done')

            if not was_added:
                logging.info(f'Adding model {self.identifier} to db')
                assert isinstance(self, Component)
                db.add(self)

        if listen:
            assert db is not None
            assert select is not None
            return self._predict_and_listen(
                X=X,
                db=db,
                select=select,
                max_chunk_size=max_chunk_size,
                **kwargs,
            )

        # TODO: tidy up this logic
        if select is not None and db is not None and db.compute.type == 'distributed':
            return self.create_predict_job(
                X,
                select=select,
                ids=ids,
                max_chunk_size=max_chunk_size,
                overwrite=overwrite,
                **kwargs,
            )(db=db, dependencies=dependencies)
        else:
            if select is not None and ids is None:
                assert db is not None
                return self._predict_with_select(
                    X=X,
                    select=select,
                    db=db,
                    in_memory=in_memory,
                    max_chunk_size=max_chunk_size,
                    overwrite=overwrite,
                    **kwargs,
                )
            elif select is not None and ids is not None:
                assert db is not None
                return self._predict_with_select_and_ids(
                    X=X,
                    select=select,
                    ids=ids,
                    db=db,
                    max_chunk_size=max_chunk_size,
                    in_memory=in_memory,
                    **kwargs,
                )
            else:
                if self.takes_context:
                    kwargs['context'] = context

                if isinstance(key, str):
                    if one:
                        assert isinstance(X, dict)
                        X = X[key]
                    else:
                        X = [r[key] for r in X]
                elif isinstance(key, list):
                    X = (
                        (
                            (X[k] for k in key)
                            if one
                            else [(r[k] for k in key) for r in X]
                        )
                        if key
                        else X,
                    )
                elif isinstance(key, dict):
                    X = (
                        (
                            (X[k] for k in key.values())
                            if one
                            else [(r[k] for k in key.values()) for r in X]
                        )
                        if key
                        else X,
                    )

                else:
                    if key is not None:
                        raise TypeError

                output = self._predict(
                    X,
                    one=one,
                    **kwargs,
                )
                if insert_to is not None:
                    msg = (
                        '`self.db` has not been set; this is necessary if'
                        ' `insert_to` is not None; use `db.add(self)`'
                    )

                    from superduperdb.base.datalayer import Datalayer

                    assert isinstance(db, Datalayer), msg
                    if isinstance(insert_to, str):
                        insert_to = db.load(
                            'table',
                            insert_to,
                        )  # type: ignore[assignment]
                    if one:
                        output = [output]

                    assert isinstance(insert_to, TableOrCollection)
                    X = [Document(X)] if one else [Document(x) for x in X]
                    inserted_ids, _ = db.execute(insert_to.insert(X))
                    inserted_ids = t.cast(t.List[t.Any], inserted_ids)
                    assert isinstance(key, str)

                    insert_to.model_update(
                        db=db,
                        model=self.identifier,
                        outputs=output,
                        key=key,
                        version=self.version,
                        ids=inserted_ids,
                        flatten=self.flatten,
                        **self.model_update_kwargs,
                    )
                return output

    async def apredict(
        self,
        X: t.Any,
        context: t.Optional[t.Dict] = None,
        one: bool = False,
        **kwargs,
    ):
        if self.takes_context:
            kwargs['context'] = context
        return await self._apredict(X, one=one, **kwargs)

    def _predict_and_listen(
        self,
        X: t.Any,
        select: CompoundSelect,
        db: Datalayer,
        in_memory: bool = True,
        max_chunk_size: t.Optional[int] = None,
        dependencies: t.Sequence[Job] = (),
        **kwargs,
    ):
        from superduperdb.components.listener import Listener

        return db.add(
            Listener(
                key=X,
                model=t.cast(Model, self),
                select=select,
                predict_kwargs={
                    **kwargs,
                    'in_memory': in_memory,
                    'max_chunk_size': max_chunk_size,
                },
            ),
            dependencies=dependencies,
        )[0]

    def _predict_with_select(
        self,
        X: t.Any,
        select: Select,
        db: Datalayer,
        max_chunk_size: t.Optional[int] = None,
        in_memory: bool = True,
        overwrite: bool = False,
        **kwargs,
    ):
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

        return self._predict_with_select_and_ids(
            X=X,
            db=db,
            ids=ids,
            select=select,
            max_chunk_size=max_chunk_size,
            in_memory=in_memory,
            **kwargs,
        )

    def _predict_with_select_and_ids(
        self,
        X: t.Any,
        db: Datalayer,
        select: Select,
        ids: t.List[str],
        in_memory: bool = True,
        max_chunk_size: t.Optional[int] = None,
        **kwargs,
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
                    **kwargs,
                )
                it += 1
            return

        X_data: t.Any
        if in_memory:
            if db is None:
                raise ValueError('db cannot be None')
            docs = list(db.execute(select.select_using_ids(ids)))
            if X == '_base':
                X_data = [r.unpack() for r in docs]
            elif isinstance(X, str):
                X_data = [MongoStyleDict(r.unpack())[X] for r in docs]
            else:
                X_data = []
                for doc in docs:
                    doc = MongoStyleDict(doc.unpack())
                    if isinstance(X, (tuple, list)):
                        X_data.append([doc[k] for k in X])
                    elif isinstance(X, dict):
                        X_data.append([doc[k] for k in X.values()])
                    else:
                        raise TypeError

        else:
            X_data = QueryDataset(
                select=select,
                ids=ids,
                fold=None,
                db=db,
                in_memory=False,
                keys=[X],
            )

        if len(X_data) > len(ids):
            raise Exception(
                'You\'ve specified more documents than unique ids;'
                f' Is it possible that {select.table_or_collection.primary_id}'
                f' isn\'t unique identifying?'
            )

        outputs = self.predict(X=X_data, one=False, **kwargs)

        if isinstance(self.datatype, DataType):
            if self.flatten:
                outputs = [
                    [self.datatype(x).encode() for x in output] for output in outputs
                ]
            else:
                outputs = [self.datatype(x).encode() for x in outputs]
        elif isinstance(self.output_schema, Schema):
            encoded_ouputs = []
            for output in outputs:
                if isinstance(output, dict):
                    encoded_ouputs.append(self.output_schema(output))
                elif self.flatten:
                    encoded_output = [self.output_schema(x) for x in output]
                    encoded_ouputs.append(encoded_output)
            outputs = encoded_ouputs if encoded_ouputs else outputs

        assert isinstance(self.version, int)

        logging.info(f'Adding {len(outputs)} model outputs to `db`')
        key = X
        if isinstance(X, (tuple, list)):
            key = ','.join(X)
        elif isinstance(X, dict):
            key = ','.join(list(X.values()))

        select.model_update(
            db=db,
            model=self.identifier,
            outputs=outputs,
            key=key,
            version=self.version,
            ids=ids,
            flatten=self.flatten,
            **self.model_update_kwargs,
        )


@public_api(stability='stable')
@dc.dataclass(kw_only=True)
class Model(_Predictor, Component):
    """Model component which wraps a model to become serializable
    {component_params}
    {_predictor_params}
    :param object: Model object, e.g. sklearn model, etc..
    :param model_to_device_method: The method to transfer the model to a device
    :param metric_values: The metric values
    :param predict_method: The method to use for prediction
    :param model_update_kwargs: The kwargs to use for model update
    :param serializer: Serializer to store model to artifact store
    :param device: The device to use
    :param preferred_devices: The preferred devices to use
    :param training_configuration: The training configuration
    :param train_X: The key of the input data to use for training
    :param train_y: The key of the target data to use for training
    :param train_select: The select to use for training
    """

    __doc__ = __doc__.format(
        component_params=Component.__doc__,
        _predictor_params=_Predictor.__doc__,
    )

    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = (
        ('object', dill_serializer),
    )

    object: t.Any
    model_to_device_method: t.Optional[str] = None
    metric_values: t.Optional[t.Dict] = dc.field(default_factory=dict)
    predict_method: t.Optional[str] = None
    model_update_kwargs: dict = dc.field(default_factory=dict)
    device: str = "cpu"
    preferred_devices: t.Union[None, t.Sequence[str]] = ("cuda", "mps", "cpu")

    training_configuration: t.Union[str, _TrainingConfiguration, None] = None
    train_X: t.Optional[str] = None
    train_y: t.Optional[str] = None
    train_select: t.Optional[CompoundSelect] = None

    type_id: t.ClassVar[str] = 'model'

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)
        self._artifact_method = None
        if self.model_to_device_method is not None:
            self._artifact_method = getattr(self, self.model_to_device_method)

    def to_call(self, X, *args, **kwargs):
        if isinstance(X, (tuple, list)):
            required_args = len(self.inputs)
            assert len(X) == required_args
            X = self.inputs.get_kwargs(X)

        elif isinstance(X, dict):
            required_args = len(self.inputs)
            assert len(X) == required_args
        else:
            X = self.inputs.get_kwargs([X])

        if self.predict_method is None:
            return self.object(*args, **X, **kwargs)
        out = getattr(self.object, self.predict_method)(*X, *args, **kwargs)
        return out

    def post_create(self, db: Datalayer) -> None:
        if isinstance(self.training_configuration, str):
            self.training_configuration = db.load(
                'training_configuration', self.training_configuration
            )  # type: ignore[assignment]
        # TODO is this necessary - should be handled by `db.add` automatically
        if isinstance(self.output_schema, Schema):
            db.add(self.output_schema)
        output_component = db.databackend.create_model_table_or_collection(self)
        if output_component is not None:
            db.add(output_component)

    # TODO - bring inside post_create
    @override
    def schedule_jobs(
        self,
        db: Datalayer,
        dependencies: t.Sequence[Job] = (),
        verbose: bool = False,
    ) -> t.Sequence[t.Any]:
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
                    metrics=self.metrics,  # type: ignore[arg-type]
                    validation_sets=self.validation_sets,
                )
            )
        if self.predict_X is not None:
            assert self.predict_select is not None
            jobs.append(
                self.predict(
                    X=self.predict_X,
                    select=self.predict_select,
                    max_chunk_size=self.predict_max_chunk_size,
                    db=db,
                    **(self.predict_kwargs or {}),
                )
            )
        return jobs

    def on_load(self, db: Datalayer) -> None:
        logging.debug(f'Calling on_load method of {self}')
        if self._artifact_method and self.preferred_devices:
            for i, device in enumerate(self.preferred_devices):
                try:
                    self._artifact_method(device)
                    self.device = device
                    return
                except Exception:
                    if i == len(self.preferred_devices) - 1:
                        raise

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

    def pre_create(self, db: Datalayer):
        # TODO this kind of thing should come from an enum component_types.datatype
        # that will make refactors etc. easier
        if isinstance(self.datatype, str):
            # ruff: noqa: E501
            self.datatype = db.load('datatype', self.datatype)  # type: ignore[assignment]

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
        prediction = self._predict([d[self.train_X] for d in mdicts])
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

    def _fit(
        self,
        X: t.Any,
        y: t.Optional[t.Any] = None,
        configuration: t.Optional[_TrainingConfiguration] = None,
        data_prefetch: bool = False,
        db: t.Optional[Datalayer] = None,
        metrics: t.Optional[t.Sequence[Metric]] = None,
        select: t.Optional[Select] = None,
        validation_sets: t.Optional[t.Sequence[t.Union[str, Dataset]]] = None,
    ):
        raise NotImplementedError

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
        validation_sets: t.Optional[t.Sequence[t.Union[str, Dataset]]] = None,
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
                    validation_sets[i] = vs.identifier

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


@wraps(_TrainingConfiguration)
def TrainingConfiguration(identifier: str, **kwargs):
    return _TrainingConfiguration(identifier=identifier, kwargs=kwargs)


@public_api(stability='beta')
@dc.dataclass(kw_only=True)
class APIModel(Component, _Predictor):
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

    @override
    def schedule_jobs(
        self,
        db: Datalayer,
        dependencies: t.Sequence[Job] = (),
        verbose: bool = False,
    ) -> t.Sequence[t.Any]:
        jobs = []
        if self.predict_X is not None:
            assert self.predict_select is not None
            jobs.append(
                self.predict(
                    X=self.predict_X,
                    select=self.predict_select,
                    max_chunk_size=self.predict_max_chunk_size,
                    db=db,
                    **(self.predict_kwargs or {}),
                )
            )
        return jobs


@public_api(stability='stable')
@dc.dataclass(kw_only=True)
class QueryModel(Component, _Predictor):
    """
    Model which can be used to query data and return those
    results as pre-computed queries.

    :param select: query used to find data (can include `like`)
    """

    select: CompoundSelect

    def schedule_jobs(
        self,
        db: Datalayer,
        dependencies: t.Sequence[Job] = (),
        verbose: bool = False,
    ) -> t.Sequence[t.Any]:
        jobs = []
        if self.predict_X is not None:
            assert self.predict_select is not None
            jobs.append(
                self.predict(
                    X=self.predict_X,
                    select=self.predict_select,
                    max_chunk_size=self.predict_max_chunk_size,
                    db=db,
                    **(self.predict_kwargs or {}),
                )
            )
        return jobs

    def _predict_one(self, X: t.Any, **kwargs):
        select = self.select.set_variables(db=self.db, X=X)
        out = self.db.execute(select)
        if self.postprocess is not None:
            return self.postprocess(out)
        return out

    def _predict(self, X: t.Any, one: bool = False, **predict_kwargs):
        if one:
            return self._predict_one(X, **predict_kwargs)
        return [self._predict_one(x, **predict_kwargs) for x in X]


@public_api(stability='stable')
@dc.dataclass(kw_only=True)
class SequentialModel(Component, _Predictor):
    """
    Sequential model component which wraps a model to become serializable

    {component_params}
    {_predictor_params}
    :param predictors: A list of predictors to use
    """

    __doc__ = __doc__.format(
        component_params=Component.__doc__,
        _predictor_params=_Predictor.__doc__,
    )
    predictors: t.List[t.Union[str, Model, APIModel]]

    @override
    def schedule_jobs(
        self,
        db: Datalayer,
        dependencies: t.Sequence[Job] = (),
        verbose: bool = False,
    ) -> t.Sequence[t.Any]:
        jobs = []
        if self.predict_X is not None:
            assert self.predict_select is not None
            jobs.append(
                self.predict(
                    X=self.predict_X,
                    select=self.predict_select,
                    max_chunk_size=self.predict_max_chunk_size,
                    db=db,
                    **(self.predict_kwargs or {}),
                )
            )
        return jobs

    def post_create(self, db: Datalayer):
        for p in self.predictors:
            if isinstance(p, str):
                continue
            p.post_create(db)
        self.on_load(db)

    def on_load(self, db: Datalayer):
        for i, p in enumerate(self.predictors):
            if isinstance(p, str):
                self.predictors[i] = db.load('model', p)  # type: ignore[call-overload]

    def _predict(self, X: t.Any, one: bool = False, **predict_kwargs):
        out = X
        for p in self.predictors:
            assert isinstance(p, _Predictor)
            out = p._predict(out, one=one, **predict_kwargs)
        return out
