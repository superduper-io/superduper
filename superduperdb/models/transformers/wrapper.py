import dataclasses as dc
import functools
import logging
import random
import re
import typing as t

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    pipeline as _pipeline,
    Pipeline as BasePipeline,
    DataCollatorWithPadding,
)

from superduperdb.core.metric import Metric
from superduperdb.core.model import Model
from superduperdb.core.artifact import Artifact
from superduperdb.datalayer.base.datalayer import Datalayer
from superduperdb.datalayer.base.query import Select
from superduperdb.core.model import _TrainingConfiguration
from superduperdb.datalayer.query_dataset import query_dataset_factory
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
    preprocess_type: str = 'tokenizer'
    task: str = 'text-classification'

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.object.artifact, BasePipeline):
            assert self.preprocess is None
            if self.preprocess_type == 'tokenizer':
                self.preprocess = self.object.artifact.tokenizer
            elif self.preprocess_type == 'feature_extractor':
                raise NotImplementedError
            elif self.preprocess_type == 'image_process':
                raise NotImplementedError
            else:
                raise NotImplementedError
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
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @functools.cached_property
    def pipeline(self):
        if self.preprocess_type == 'tokenizer':
            return _pipeline(
                self.task,
                model=self.object.artifact,
                tokenizer=self.preprocess.artifact,
            )
        elif self.preprocess_type == 'feature_extractor':
            raise NotImplementedError
        elif self.preprocess_type == 'image_preprocessor':
            raise NotImplementedError
        else:
            raise NotImplementedError

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
            if self.preprocess_type == 'tokenizer':
                r.update(**self.preprocess.artifact(text, truncation=True))
            else:
                raise NotImplementedError
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

    def _fit(  # type: ignore[override,return]
        self,
        X: str,
        y: str,
        select: t.Optional[Select] = None,
        db: t.Optional[Datalayer] = None,
        configuration: t.Optional[_TrainingConfiguration] = None,
        validation_sets: t.Optional[t.List[str]] = None,
        metrics: t.Optional[t.List[Metric]] = None,
        data_prefetch: bool = False,
        prefetch_size: int = _DEFAULT_PREFETCH_SIZE,
        **kwargs,
    ) -> t.Optional[t.Dict[str, t.Any]]:
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

        trainer = TrainerWithSaving(
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
        if not one:
            if self.preprocess_type == 'tokenizer':
                out = self.pipeline(X, truncation=True, **kwargs)
            else:
                out = self.pipeline(X, **kwargs)
            out = [r['label'] for r in out]
            for i, p in enumerate(out):
                if re.match(r'^LABEL_[0-9]+', p):
                    out[i] = int(p[6:])
        else:
            out = self.pipeline(X, **kwargs)
            out = out[0]['label']
            if re.match(r'^LABEL_[0-9]+', out):
                out = int(out[6:])
        return out


class PreprocessFunction:
    def __init__(self, preprocess, key: str = '', **kwargs):
        self.preprocess = preprocess
        self.kwargs = kwargs
        self.key = key

    def __call__(self, input):
        x = input[self.key]
        input.update(**self.preprocess(x, **self.kwargs))
        return input


class TrainerWithSaving(Trainer):
    def __init__(self, custom_saver=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_saver = custom_saver

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics=metrics)
        logging.info('Saving model...')
        if self.custom_saver:
            self.custom_saver()
