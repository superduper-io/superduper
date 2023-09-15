import dataclasses as dc
import functools
import logging
import random
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

from superduperdb.container.artifact import Artifact
from superduperdb.container.metric import Metric
from superduperdb.container.model import Model, _TrainingConfiguration
from superduperdb.db.base.db import DB
from superduperdb.db.base.query import Select
from superduperdb.db.query_dataset import query_dataset_factory
from superduperdb.misc.special_dicts import MongoStyleDict

_DEFAULT_PREFETCH_SIZE: int = 100


@functools.wraps(TrainingArguments)
def TransformersTrainerConfiguration(identifier: str, *args, **kwargs):
    if 'output_dir' in kwargs:
        cfg = TrainingArguments(**kwargs)
    else:
        cfg = TrainingArguments(output_dir=args[0], **kwargs)
    return _TrainingConfiguration(identifier=identifier, kwargs=cfg.to_dict())


@dc.dataclass
class Pipeline(Model):
    """A wrapper for ``transformers.Pipeline``

    :param preprocess_type: The type of preprocessing to use {'tokenizer'}
    :param preprocess_kwargs: The type of preprocessing to use. Currently only
    :param postprocess_kwargs: The type of postprocessing to use.
    :param task: The task to use for the pipeline.
    """

    preprocess_type: str = 'tokenizer'
    preprocess_kwargs: t.Dict[str, t.Any] = dc.field(default_factory=dict)
    postprocess_kwargs: t.Dict[str, t.Any] = dc.field(default_factory=dict)
    task: str = 'text-classification'

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.object.artifact, BasePipeline):
            assert self.preprocess is None
            if self.preprocess_type == 'tokenizer':
                self.preprocess = self.object.artifact.tokenizer
            else:
                raise NotImplementedError(
                    'Only tokenizer is supported for now in pipeline mode'
                )
            self.object = Artifact(artifact=self.object.artifact.model)
            self.task = self.object.artifact.task
        if (
            self.collate_fn is None
            and self.preprocess is not None
            and self.preprocess_type == 'tokenizer'
        ):
            self.collate_fn = Artifact(
                DataCollatorWithPadding(self.preprocess.artifact),
                hash=random.randrange(1000000),
            )
        if not self.device:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @functools.cached_property
    def pipeline(self):
        if self.preprocess_type == 'tokenizer':
            return _pipeline(
                self.task,
                model=self.object.artifact,
                tokenizer=self.preprocess.artifact,
            )
        else:
            warnings.warn('Only tokenizer is supported for now in pipeline mode')

    def _predict_with_preprocess_object_post(self, X, **kwargs):
        X = self.preprocess.artifact(X, **self.preprocess_kwargs)
        X = getattr(self.object.artifact, self.predict_method)(**X, **kwargs)
        X = getattr(self, 'postprocess', Artifact(lambda x: x)).artifact(
            X, **self.postprocess_kwargs
        )
        return X

    @functools.cached_property
    def training_arguments(self):
        return TrainingArguments(**self.configuration.kwargs)

    def _get_data(
        self,
        db,
        valid_data=None,
        data_prefetch: bool = False,
        prefetch_size: int = 100,
        X: str = '',
    ):
        def transform_function(r):
            text = r[X]
            r.update(**self.preprocess.artifact(text, **self.preprocess_kwargs))
            return r

        train_data = query_dataset_factory(
            select=self.training_select,
            keys=self.training_keys,
            fold='train',
            transform=transform_function,
            data_prefetch=data_prefetch,
            prefetch_size=prefetch_size,
            db=db,
        )
        valid_data = query_dataset_factory(
            select=self.training_select,
            keys=self.training_keys,
            fold='valid',
            transform=transform_function,
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
        db: t.Optional[DB] = None,
        metrics: t.Optional[t.Sequence[Metric]] = None,
        prefetch_size: int = _DEFAULT_PREFETCH_SIZE,
        select: t.Optional[Select] = None,
        validation_sets: t.Optional[t.Sequence[str]] = None,
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

        if isinstance(X, str):
            train_data, valid_data = self._get_data(
                db,
                valid_data=validation_sets,
                data_prefetch=data_prefetch,
                prefetch_size=prefetch_size,
                X=X,
            )

        def compute_metrics(eval_pred):
            output = {}
            for vs in validation_sets:
                vs = db.load('dataset', vs)
                unpacked = [MongoStyleDict(r.unpack()) for r in vs.data]
                predictions = self._predict([r[X] for r in unpacked])
                targets = [r[y] for r in unpacked]
                for m in metrics:
                    output[f'{vs.identifier}/{m.identifier}'] = m(predictions, targets)
            self.append_metrics(output)
            return output

        assert isinstance(self.collate_fn, Artifact)
        assert db is not None
        trainer = _TrainerWithSaving(
            model=self.object.artifact,
            args=self.training_arguments,
            train_dataset=train_data,
            eval_dataset=valid_data,
            data_collator=self.collate_fn.artifact,
            custom_saver=lambda: db.replace(self, upsert=True),
            compute_metrics=compute_metrics,
            **kwargs,
        )
        trainer.train()

    def _predict(self, X, one: bool = False, **kwargs):
        if self.pipeline is not None:
            out = self.pipeline(X, **self.preprocess_kwargs, **kwargs)
            out = [r['label'] for r in out]
            for i, p in enumerate(out):
                if re.match(r'^LABEL_[0-9]+', p):
                    out[i] = int(p[6:])
        else:
            out = self._predict_with_preprocess_object_post(X, **kwargs)
        if one:
            return out[0]
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
