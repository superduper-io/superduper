import dataclasses as dc
import functools
import re
import typing as t
import warnings

from transformers import (
    DataCollatorWithPadding,
    Pipeline as BasePipeline,
    Trainer,
    TrainingArguments,
    pipeline as _pipeline,
)

from superduperdb import logging
from superduperdb.backends.base.query import Select
from superduperdb.backends.query_dataset import QueryDataset, query_dataset_factory
from superduperdb.base.datalayer import Datalayer
from superduperdb.components.dataset import Dataset
from superduperdb.components.metric import Metric
from superduperdb.components.model import (
    Signature,
    _DeviceManaged,
    _Fittable,
    _Predictor,
    _TrainingConfiguration,
)
from superduperdb.misc.special_dicts import MongoStyleDict

_DEFAULT_PREFETCH_SIZE: int = 100


@functools.wraps(TrainingArguments)
def TransformersTrainerConfiguration(identifier: str, *args, **kwargs):
    if 'output_dir' in kwargs:
        cfg = TrainingArguments(**kwargs)
    else:
        cfg = TrainingArguments(output_dir=args[0], **kwargs)
    return _TrainingConfiguration(identifier=identifier, kwargs=cfg.to_dict())


# TODO refactor
@dc.dataclass
class Pipeline(_Predictor, _Fittable, _DeviceManaged):
    """A wrapper for ``transformers.Pipeline``

    :param object: The object
    :param postprocess: The postprocessor
    :param preprocess_type: The type of preprocessing to use {'tokenizer'}
    :param preprocess_kwargs: The type of preprocessing to use. Currently only
    :param postprocessor: The postprocessing function
    :param postprocess_kwargs: The type of postprocessing to use.
    :param task: The task to use for the pipeline.
    """

    signature: t.ClassVar[str] = Signature.singleton
    object: t.Optional[t.Callable] = None
    preprocess: t.Optional[t.Callable] = None
    preprocess_type: str = 'tokenizer'
    preprocess_kwargs: t.Dict[str, t.Any] = dc.field(default_factory=dict)
    postprocess: t.Optional[t.Callable] = None
    postprocess_kwargs: t.Dict[str, t.Any] = dc.field(default_factory=dict)
    collate_fn: t.Optional[t.Callable] = None
    task: str = 'text-classification'

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)
        if isinstance(self.object, BasePipeline):
            assert self.preprocess is None
            if self.preprocess_type == 'tokenizer':
                self.preprocess = self.object.tokenizer
            else:
                raise NotImplementedError(
                    'Only tokenizer is supported for now in pipeline mode'
                )

            if (
                self.collate_fn is None
                and self.preprocess is not None
                and self.preprocess_type == 'tokenizer'
            ):
                self.collate_fn = DataCollatorWithPadding(self.preprocess)
            self.object = self.object.model
            self.task = self.object.task
        if (
            self.collate_fn is None
            and self.preprocess is not None
            and self.preprocess_type == 'tokenizer'
        ):
            self.collate_fn = DataCollatorWithPadding(self.preprocess)
        if not self.device:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @functools.cached_property
    def pipeline(self):
        if self.preprocess_type == 'tokenizer':
            return _pipeline(
                self.task,
                model=self.object,
                tokenizer=self.preprocess,
            )
        else:
            warnings.warn('Only tokenizer is supported for now in pipeline mode')

    # TODO very confusing...
    def _predict_with_preprocess_object_post(self, X, **kwargs):
        X = self.preprocess(X, **self.preprocess_kwargs)
        X = getattr(self.object, self.predict_method)(**X, **kwargs)
        postprocess = self.postprocess or (lambda x: x)
        X = postprocess(X, **self.postprocess_kwargs)
        return X

    @functools.cached_property
    def training_arguments(self):
        return TrainingArguments(**self.configuration.kwargs)

    def _get_data(
        self,
        db,
        X: str,
        valid_data=None,
        data_prefetch: bool = False,
        prefetch_size: int = 100,
    ):
        def preprocess(r):
            if self.preprocess:
                r.update(self.preprocess(r[X], **self.preprocess_kwargs))
            return r

        train_data = query_dataset_factory(
            select=self.training_select,
            fold='train',
            transform=preprocess,
            data_prefetch=data_prefetch,
            prefetch_size=prefetch_size,
            db=db,
        )
        valid_data = query_dataset_factory(
            select=self.training_select,
            fold='valid',
            transform=preprocess,
            data_prefetch=data_prefetch,
            prefetch_size=prefetch_size,
            db=db,
        )
        return train_data, valid_data

    def _fit(  # type: ignore[override]
        self,
        X: str,
        y: str,
        configuration: t.Optional[_TrainingConfiguration] = None,
        data_prefetch: bool = False,
        db: t.Optional[Datalayer] = None,
        metrics: t.Optional[t.Sequence[Metric]] = None,
        prefetch_size: int = _DEFAULT_PREFETCH_SIZE,
        select: t.Optional[Select] = None,
        validation_sets: t.Optional[t.Sequence[Dataset]] = None,
        **kwargs,
    ) -> None:
        if configuration is not None:
            self.configuration = configuration
        if select is not None:
            self.training_select = select
        if validation_sets is not None:
            self.validation_sets = validation_sets
        if metrics is not None:
            self.metrics = metrics
        self.train_X = X
        self.train_y = y

        train_data, valid_data = self._get_data(
            db,
            X,
            valid_data=validation_sets,
            data_prefetch=data_prefetch,
            prefetch_size=prefetch_size,
        )

        def compute_metrics(eval_pred):
            output = {}
            for vs in validation_sets:
                vs = db.load('dataset', vs)
                unpacked = [MongoStyleDict(r.unpack()) for r in vs.data]
                predictions = self.predict([r[X] for r in unpacked])
                targets = [r[y] for r in unpacked]
                for m in metrics:
                    output[f'{vs.identifier}/{m.identifier}'] = m(predictions, targets)
            self.append_metrics(output)
            return output

        assert self.collate_fn
        assert db is not None
        assert self.object
        trainer = _TrainerWithSaving(
            model=self.object,
            args=self.training_arguments,
            train_dataset=train_data,
            eval_dataset=valid_data,
            data_collator=self.collate_fn,
            custom_saver=lambda: db.replace(self, upsert=True),
            compute_metrics=compute_metrics,
            **kwargs,
        )
        trainer.train()

    def predict_one(self, X: t.Any):
        return self.predict([X])[0]

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        if self.pipeline is not None:
            X = [dataset[i] for i in range(len(dataset))]
            out = self.pipeline(X, **self.preprocess_kwargs, **self.predict_kwargs)
            out = [r['label'] for r in out]
            for i, p in enumerate(out):
                if re.match(r'^LABEL_[0-9]+', p):
                    out[i] = int(p[6:])
        else:
            out = self._predict_with_preprocess_object_post(
                dataset, **self.predict_kwargs
            )
        return out


class _TrainerWithSaving(Trainer):
    def __init__(self, custom_saver=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_saver = custom_saver

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics=metrics)
        logging.info('Saving model...')
        if self.custom_saver:
            self.custom_saver()
