from __future__ import annotations

import dataclasses as dc
import multiprocessing
import typing as t
from abc import abstractmethod
from functools import wraps

import tqdm
from numpy import ndarray
from sklearn.pipeline import Pipeline

from superduperdb import logging
from superduperdb.backends.base.metadata import NonExistentMetadataError
from superduperdb.backends.base.query import CompoundSelect, Select
from superduperdb.backends.ibis.field_types import FieldType
from superduperdb.backends.ibis.query import IbisCompoundSelect, Table
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.base.artifact import Artifact
from superduperdb.base.serializable import Serializable
from superduperdb.components.component import Component
from superduperdb.components.encoder import Encoder
from superduperdb.components.metric import Metric
from superduperdb.components.schema import Schema
from superduperdb.jobs.job import ComponentJob, Job
from superduperdb.misc.special_dicts import MongoStyleDict

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.dataset import Dataset

EncoderArg = t.Union[Encoder, FieldType, str, None]
ObjectsArg = t.Sequence[t.Union[t.Any, Artifact]]


class _to_call:
    def __init__(self, callable, **kwargs):
        self.callable = callable
        self.kwargs = kwargs

    def __call__(self, X):
        return self.callable(X, **self.kwargs)


@dc.dataclass
class _TrainingConfiguration(Component):
    """
    Training configuration object, containing all settings necessary for a particular
    learning-task use-case to be serialized and initiated. The object is ``callable``
    and returns a class which may be invoked to apply training.

    :param identifier: Unique identifier of configuration
    :param **kwargs: Key-values pairs, the variables which configure training.
    """

    identifier: str
    kwargs: t.Optional[t.Dict] = None

    version: t.Optional[int] = None

    type_id: t.ClassVar[str] = 'training_configuration'

    def get(self, k, default=None):
        try:
            return getattr(self, k)
        except AttributeError:
            return self.kwargs.get(k, default)


class Predictor:
    """
    Mixin class for components which can predict.

    :param identifier: Unique identifier of model
    :param encoder: Encoder instance (optional)
    :param output_schema: Output schema (mapping of encoders) (optional)
    :param flatten: Flatten the model outputs
    :param preprocess: Preprocess function (optional)
    :param postprocess: Postprocess function (optional)
    :param collate_fn: Collate function (optional)
    :param batch_predict: Whether to batch predict (optional)
    :param takes_context: Whether the model takes context into account (optional)
    :param to_call: The method to use for prediction (optional)
    :param model_update_kwargs: The kwargs to use for model update (optional)
    """

    identifier: str
    encoder: EncoderArg
    output_schema: t.Optional[t.Union[Schema, dict]] = None
    flatten: bool = False
    preprocess: t.Union[t.Callable, Artifact, None] = None
    postprocess: t.Union[t.Callable, Artifact, None] = None
    collate_fn: t.Union[t.Callable, Artifact, None] = None
    batch_predict: bool
    takes_context: bool
    to_call: t.Callable
    model_update_kwargs: t.Dict

    version: t.Optional[int] = None
    type_id: t.ClassVar[str] = 'model'

    def create_predict_job(
        self,
        X: str,
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
                'select': select.serialize() if select else None,
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
        if isinstance(self.preprocess, Artifact):
            X = self.preprocess.artifact(X)
        elif self.preprocess is not None:
            raise ValueError('Bad preprocess')

        output = self.to_call(X, **kwargs)

        if isinstance(self.postprocess, Artifact):
            output = self.postprocess.artifact(output)
        elif self.postprocess is not None:
            raise ValueError('Bad postprocess')

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

    def _predict(
        self, X: t.Any, one: bool = False, **predict_kwargs
    ) -> t.Union[ndarray, int, t.Sequence[int]]:
        if one:
            return self._predict_one(X)

        if isinstance(self.preprocess, Artifact):
            X = [self.preprocess.artifact(i) for i in X]
        elif self.preprocess is not None:
            raise ValueError('Bad preprocess')

        if isinstance(self.collate_fn, Artifact):
            raise ValueError('Bad collate function')
        elif self.collate_fn is not None:
            X = self.collate_fn(X)

        outputs = self._forward(X, **predict_kwargs)

        if isinstance(self.postprocess, Artifact):
            outputs = [self.postprocess.artifact(o) for o in outputs]
        elif self.postprocess is not None:
            raise ValueError('Bad postprocess')

        return outputs

    def predict(
        self,
        X: t.Any,
        db: t.Optional[Datalayer] = None,
        select: t.Optional[Select] = None,
        ids: t.Optional[t.List[str]] = None,
        max_chunk_size: t.Optional[int] = None,
        dependencies: t.Sequence[Job] = (),
        listen: bool = False,
        one: bool = False,
        context: t.Optional[t.Dict] = None,
        in_memory: bool = True,
        overwrite: bool = False,
        **kwargs,
    ) -> t.Any:
        if one:
            assert db is None, 'db must be None when ``one=True`` (direct call)'

        if isinstance(select, dict):
            select = Serializable.deserialize(select)

        if isinstance(select, Table):
            select = select.to_query()

        if db is not None:
            if isinstance(select, IbisCompoundSelect):
                try:
                    _ = db.metadata.get_query(str(hash(select)))
                except NonExistentMetadataError:
                    logging.info(f'Query {select} not found in metadata, adding...')
                    db.metadata.add_query(select, self.identifier)
                    logging.info('Done')
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

        if db is not None and db.compute.type == 'distributed':
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
                return self._predict(X, one=one, **kwargs)

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

        for r in tqdm.tqdm(db.execute(query)):
            ids.append(str(r[db.databackend.id_field]))

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
            if X != '_base':
                X_data = [MongoStyleDict(r.unpack())[X] for r in docs]
            else:
                X_data = [r.unpack() for r in docs]
        else:
            X_data = QueryDataset(
                select=select,
                ids=ids,
                fold=None,
                db=db,
                in_memory=False,
                keys=[X],
            )

        outputs = self.predict(X=X_data, one=False, **kwargs)

        if isinstance(self.encoder, Encoder):
            if self.flatten:
                outputs = [
                    [self.encoder(x).encode() for x in output] for output in outputs
                ]
            else:
                outputs = [self.encoder(x).encode() for x in outputs]
        elif isinstance(self.output_schema, Schema):
            encoded_ouputs = []
            for output in outputs:
                if isinstance(output, dict):
                    encoded_ouputs.append(self.output_schema.encode(output))
                elif self.flatten:
                    encoded_output = [self.output_schema.encode(x) for x in output]
                    encoded_ouputs.append(encoded_output)
            outputs = encoded_ouputs if encoded_ouputs else outputs

        assert isinstance(self.version, int)
        select.model_update(
            db=db,
            model=self.identifier,
            outputs=outputs,
            key=X,
            version=self.version,
            ids=ids,
            flatten=self.flatten,
            **self.model_update_kwargs,
        )


@dc.dataclass
class Model(Component, Predictor):
    """Model component which wraps a model to become serializable

    :param identifier: Unique identifier of model
    :param object: Model object, e.g. sklearn model, etc..
    :param encoder: Encoder instance (optional)
    :param flatten: Flatten the model outputs
    :param output_schema: Output schema (mapping of encoders) (optional)
    :param preprocess: Preprocess function (optional)
    :param postprocess: Postprocess function (optional)
    :param collate_fn: Collate function (optional)
    :param metrics: Metrics to use (optional)
    :param predict_method: The method to use for prediction (optional)
    :param model_to_device_method: The method to transfer the model to a device
    :param batch_predict: Whether to batch predict (optional)
    :param takes_context: Whether the model takes context into account (optional)
    :param train_X: The key of the input data to use for training (optional)
    :param train_y: The key of the target data to use for training (optional)
    :param training_select: The select to use for training (optional)
    :param metric_values: The metric values (optional)
    :param training_configuration: The training configuration (optional)
    :param model_update_kwargs: The kwargs to use for model update (optional)
    :param serializer: Serializer to store model to artifact store (optional)
    :param device: The device to use (optional)
    :param preferred_devices: The preferred devices to use (optional)
    """

    identifier: str
    object: t.Union[Artifact, t.Any]
    flatten: bool = False
    output_schema: t.Optional[t.Union[Schema, dict]] = None
    encoder: EncoderArg = None
    preprocess: t.Union[t.Callable, Artifact, None] = None
    postprocess: t.Union[t.Callable, Artifact, None] = None
    collate_fn: t.Union[t.Callable, Artifact, None] = None
    metrics: t.Sequence[t.Union[str, Metric, None]] = ()
    predict_method: t.Optional[str] = None
    model_to_device_method: t.Optional[str] = None
    batch_predict: bool = False
    takes_context: bool = False
    train_X: t.Optional[str] = None
    train_y: t.Optional[str] = None
    training_select: t.Union[Select, None] = None
    metric_values: t.Optional[t.Dict] = dc.field(default_factory=dict)
    training_configuration: t.Union[str, _TrainingConfiguration, None] = None
    model_update_kwargs: dict = dc.field(default_factory=dict)
    serializer: str = 'dill'
    device: str = "cpu"
    preferred_devices: t.Union[None, t.Sequence[str]] = ("cuda", "mps", "cpu")

    # Don't set these manually
    version: t.Optional[int] = None

    artifact_attributes: t.ClassVar[t.Sequence[str]] = ['object']
    type_id: t.ClassVar[str] = 'model'

    def __post_init__(self):
        if not isinstance(self.object, Artifact):
            self.object = Artifact(artifact=self.object, serializer=self.serializer)
        if self.preprocess and not isinstance(self.preprocess, Artifact):
            self.preprocess = Artifact(artifact=self.preprocess)
        if self.postprocess and not isinstance(self.postprocess, Artifact):
            self.postprocess = Artifact(artifact=self.postprocess)
        if self.predict_method is None:
            self.to_call = self.object.artifact
        else:
            self.to_call = getattr(self.object.artifact, self.predict_method)

        self._artifact_method = None

        if self.model_to_device_method is not None:
            self._artifact_method = getattr(self, self.model_to_device_method)

    def post_create(self, db: Datalayer) -> None:
        if isinstance(self.output_schema, Schema):
            db.add(self.output_schema)
        output_component = db.databackend.create_model_table_or_collection(self)
        if output_component is not None:
            db.add(output_component)

    def on_load(self, db: Datalayer) -> None:
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
    def child_components(self) -> t.Sequence[t.Tuple[str, str]]:
        out = []
        if isinstance(self.encoder, Encoder):
            out.append(('encoder', 'encoder'))
        if self.training_configuration is not None:
            out.append(('training_configuration', 'training_configuration'))
        return out

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
        if isinstance(self.encoder, str):
            self.encoder = db.load('encoder', self.encoder)  # type: ignore[assignment]

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
                'select': select.serialize() if select else None,
                **kwargs,
            },
        )

    @abstractmethod
    def _fit(
        self,
        X: t.Any,
        y: t.Any = None,
        configuration: t.Optional[_TrainingConfiguration] = None,
        data_prefetch: bool = False,
        db: t.Optional[Datalayer] = None,
        dependencies: t.Sequence[Job] = (),
        metrics: t.Optional[t.Sequence[Metric]] = None,
        select: t.Optional[Select] = None,
        validation_sets: t.Optional[t.Sequence[t.Union[str, Dataset]]] = None,
    ):
        pass

    def fit(
        self,
        X: t.Any,
        y: t.Any = None,
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
            select = Serializable.deserialize(select)

        if validation_sets:
            from superduperdb.components.dataset import Dataset

            validation_sets = list(validation_sets)
            for i, vs in enumerate(validation_sets):
                if isinstance(vs, Dataset):
                    assert db is not None
                    db.add(vs)
                    validation_sets[i] = vs.identifier

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


@dc.dataclass
class APIModel(Component, Predictor):
    '''
    A Component for representing api models like openai, cohere etc
    :param model: The model to use, e.g. ``'text-embedding-ada-002'``.
    :param identifier: The identifier to use, e.g. ``'my-model'``.
    :param version: The version to use, e.g. ``0`` (leave empty)
    :param takes_context: Whether the model takes context into account.
    :param encoder: The encoder identifier.
    '''

    model: str
    identifier: str = ''
    version: t.Optional[int] = None
    takes_context: bool = False
    encoder: t.Union[FieldType, Encoder, str, None] = None
    model_update_kwargs: dict = dc.field(default_factory=dict)

    def post_create(self, db: Datalayer) -> None:
        if isinstance(self.output_schema, Schema):
            db.add(self.output_schema)
        output_component = db.databackend.create_model_table_or_collection(self)
        if output_component is not None:
            db.add(output_component)

    @property
    def child_components(self):
        if isinstance(self.encoder, Encoder):
            return [('encoder', 'encoder')]
        return []
