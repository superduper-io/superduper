from __future__ import annotations

import dataclasses as dc
import logging
import multiprocessing
import typing as t
from functools import wraps

import tqdm
from dask.distributed import Future
from numpy import ndarray
from sklearn.pipeline import Pipeline

import superduperdb as s
from superduperdb.container.artifact import Artifact
from superduperdb.container.component import Component
from superduperdb.container.dataset import Dataset
from superduperdb.container.encoder import Encoder
from superduperdb.container.job import ComponentJob, Job
from superduperdb.container.metric import Metric
from superduperdb.container.schema import Schema
from superduperdb.container.serializable import Serializable
from superduperdb.db.base.query import Select
from superduperdb.db.query_dataset import QueryDataset
from superduperdb.misc.special_dicts import MongoStyleDict

if t.TYPE_CHECKING:
    from superduperdb.db.base.db import DB

EncoderArg = t.Union[Encoder, str, None]
ObjectsArg = t.Sequence[t.Union[t.Any, Artifact]]
DataArg = t.Optional[t.Union[str, t.Sequence[str]]]


@dc.dataclass
class _TrainingConfiguration(Component):
    """
    Training configuration object, containing all settings necessary for a particular
    learning-task use-case to be serialized and initiated. The object is ``callable``
    and returns a class which may be invoked to apply training.

    :param identifier: Unique identifier of configuration
    :param **kwargs: Key-values pairs, the variables which configure training.
    :param version: Version number of the configuration
    """

    identifier: str
    kwargs: t.Optional[t.Dict] = None
    version: t.Optional[int] = None

    type_id: t.ClassVar[str] = 'training_configuration'

    def get(self, k, default=None):
        try:
            return getattr(self, k)
        except AttributeError:
            return self.kwargs.get(k, default=default)


class PredictMixin:
    """
    Mixin class for components which can predict.

    :param identifier: Unique identifier of model
    :param encoder: Encoder instance (optional)
    :param preprocess: Preprocess function (optional)
    :param postprocess: Postprocess function (optional)
    :param collate_fn: Collate function (optional)
    :param batch_predict: Whether to batch predict (optional)
    :param takes_context: Whether the model takes context into account (optional)
    :param to_call: The method to use for prediction (optional)
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
    model_update_kwargs: dict

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
                'distributed': False,
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

    def _forward(self, X: t.Sequence[int], num_workers: int = 0) -> t.Sequence[int]:
        if self.batch_predict:
            return self.to_call(X)

        outputs = []
        if num_workers:
            pool = multiprocessing.Pool(processes=num_workers)
            for r in pool.map(self.to_call, X):
                outputs.append(r)
            pool.close()
            pool.join()
        else:
            for r in X:
                outputs.append(self.to_call(r))
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
        db: t.Optional[DB] = None,
        select: t.Optional[Select] = None,
        distributed: t.Optional[bool] = None,
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

        if db is not None:
            logging.info(f'Adding model {self.identifier} to db')
            assert isinstance(self, Component)
            db.add(self)
            logging.info('Done.')

        if distributed is None:
            distributed = s.CFG.cluster.distributed

        if listen:
            assert db is not None
            assert select is not None
            return self._predict_and_listen(X=X, db=db, select=select, **kwargs)

        if distributed:
            return self.create_predict_job(
                X,
                select=select,
                ids=ids,
                max_chunk_size=max_chunk_size,
                overwrite=overwrite,
                **kwargs,
            )(db=db, distributed=distributed, dependencies=dependencies)
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
        select: Select,
        db: DB,
        in_memory: bool = True,
        max_chunk_size: t.Optional[int] = None,
        dependencies: t.Sequence[Job] = (),
        **kwargs,
    ):
        from superduperdb.container.listener import Listener

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
        )

    def _predict_with_select(
        self,
        X: t.Any,
        select: Select,
        db: DB,
        max_chunk_size: t.Optional[int] = None,
        in_memory: bool = True,
        overwrite: bool = False,
        **kwargs,
    ):
        ids = []
        if overwrite:
            query = select.select_ids
        else:
            query = select.select_ids_of_missing_outputs(key=X, model=self.identifier)

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
        db: DB,
        select: Select,
        ids: t.List[str],
        in_memory: bool = True,
        max_chunk_size: t.Optional[int] = None,
        **kwargs,
    ):
        if max_chunk_size is not None:
            it = 0
            for i in range(0, len(ids), max_chunk_size):
                print(f'Computing chunk {it}/{int(len(ids) / max_chunk_size)}')
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

        outputs = self.predict(X=X_data, one=False, distributed=False, **kwargs)

        if self.flatten:
            assert all([isinstance(x, (list, tuple)) for x in outputs])

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

        select.model_update(
            db=db,
            model=self.identifier,
            outputs=outputs,
            key=X,
            ids=ids,
            document_embedded=self.model_update_kwargs.get('document_embedded', True),
            flatten=self.flatten,
        )
        return


@dc.dataclass
class Model(Component, PredictMixin):
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
    :param takes_context: Whether the model takes context into account (optional)"""

    identifier: str
    object: t.Union[Artifact, t.Any]
    encoder: t.Any = None
    flatten: bool = False
    output_schema: t.Optional[t.Union[Schema, dict]] = None
    preprocess: t.Union[t.Callable, Artifact, None] = None
    postprocess: t.Union[t.Callable, Artifact, None] = None
    collate_fn: t.Union[t.Callable, Artifact, None] = None
    metrics: t.Sequence[t.Union[str, Metric, None]] = ()
    predict_method: t.Optional[str] = None
    model_to_device_method: t.Optional[str] = None
    batch_predict: bool = False
    takes_context: bool = False
    train_X: DataArg = None
    train_y: DataArg = None
    training_select: t.Union[Select, None] = None
    metric_values: t.Optional[t.Dict] = dc.field(default_factory=dict)
    training_configuration: t.Union[str, _TrainingConfiguration, None] = None
    model_update_kwargs: dict = dc.field(default_factory=dict)

    version: t.Optional[int] = None
    future: t.Optional[Future] = None
    device: str = "cpu"

    # TODO: handle situation with multiple GPUs
    preferred_devices: t.Union[None, t.Sequence[str]] = ("cuda", "mps", "cpu")

    artifacts: t.ClassVar[t.Sequence[str]] = ['object']

    type_id: t.ClassVar[str] = 'model'

    def __post_init__(self):
        if not isinstance(self.object, Artifact):
            self.object = Artifact(artifact=self.object)
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

    def on_load(self, db: DB) -> None:
        if self._artifact_method and self.preferred_devices:
            for i, device in enumerate(self.preferred_devices):
                try:
                    self._artifact_method(device)
                    self.device = device
                    return
                except Exception:
                    if i == len(self.preferred_devices) - 1:
                        raise
        if isinstance(self.output_schema, Schema):
            db.add(self.output_schema)

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

    def on_create(self, db: DB):
        if isinstance(self.encoder, str):
            self.encoder = db.load('encoder', self.encoder)
        # TODO: check if output table should be created
        db.create_output_table(self)

    def _validate(
        self, db: DB, validation_set: t.Union[Dataset, str], metrics: t.Sequence[Metric]
    ):
        if isinstance(validation_set, str):
            validation_set = t.cast(Dataset, db.load('dataset', validation_set))

        mdicts = [MongoStyleDict(r.unpack()) for r in validation_set.data]
        # TOOD: self.train_X, self.train_y are sequences of strings: this can't work
        prediction_X = self._predict(
            [d[self.train_X] for d in mdicts]  # type: ignore[index]
        )
        prediction_y = self._predict(
            [d[self.train_X] for d in mdicts]  # type: ignore[index]
        )
        assert isinstance(prediction_X, list)
        assert isinstance(prediction_y, list)
        assert all(isinstance(i, int) for i in prediction_X + prediction_y)
        results = {}

        for m in metrics:
            out = m(prediction_X, prediction_y)
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
                'distributed': False,
                'select': select.serialize() if select else None,
                **kwargs,
            },
        )

    def _fit(
        self,
        X: t.Any,
        y: t.Any = None,
        configuration: t.Optional[_TrainingConfiguration] = None,
        data_prefetch: bool = False,
        db: t.Optional[DB] = None,
        dependencies: t.Sequence[Job] = (),
        metrics: t.Optional[t.Sequence[Metric]] = None,
        select: t.Optional[Select] = None,
        validation_sets: t.Optional[t.Sequence[t.Union[str, Dataset]]] = None,
    ):
        raise NotImplementedError

    def fit(
        self,
        X: t.Any,
        y: t.Any = None,
        configuration: t.Optional[_TrainingConfiguration] = None,
        data_prefetch: bool = False,
        db: t.Optional[DB] = None,
        dependencies: t.Sequence[Job] = (),
        distributed: t.Optional[bool] = None,
        metrics: t.Optional[t.Sequence[Metric]] = None,
        select: t.Optional[Select] = None,
        validation_sets: t.Optional[t.Sequence[t.Union[str, Dataset]]] = None,
        **kwargs,
    ) -> t.Optional[Pipeline]:
        if isinstance(select, dict):
            select = Serializable.deserialize(select)

        if validation_sets:
            validation_sets = list(validation_sets)
            for i, vs in enumerate(validation_sets):
                if isinstance(vs, Dataset):
                    assert db is not None
                    db.add(vs)
                    validation_sets[i] = vs.identifier

        if db is not None:
            db.add(self)

        if distributed is None:
            distributed = s.CFG.cluster.distributed

        if distributed:
            return self.create_fit_job(
                X,
                select=select,
                y=y,
                **kwargs,
            )(db=db, distributed=True, dependencies=dependencies)
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
