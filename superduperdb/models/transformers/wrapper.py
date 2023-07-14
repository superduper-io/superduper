import dataclasses as dc
import functools
import typing as t

import torch

from superduperdb.core.metric import Metric
from superduperdb.core.model import Model
from transformers import (
    TrainingArguments,
    Trainer,
)
from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.datalayer.base.query import Select
from superduperdb.core.model import _TrainingConfiguration
from superduperdb.datalayer.query_dataset import CachedQueryDataset


@functools.wraps(TrainingArguments)
def TransformersTrainerConfiguration(identifier: str, *args, **kwargs):
    if 'output_dir' in kwargs:
        cfg = TrainingArguments(**kwargs)
    else:
        cfg = TrainingArguments(output_dir=args[0], **kwargs)
    return _TrainingConfiguration(identifier=identifier, kwargs=cfg.to_dict())


@dc.dataclass
class Pipeline(Model):
    _DEFAULT_BATCH_SIZE = 8
    data_collator: t.Any = None
    tokenizer: t.Optional[t.Callable] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self, db):
        self.object.to(self.device)
        super().__post_init__(db)

    @property
    def pipeline(self):
        return self.object.a

    @functools.cached_property
    def training_arguments(self):
        return TrainingArguments(**self.configuration.kwargs)

    def _get_data(self, db, valid_data=None):
        tokenizing_function = TokenizingFunction(self.tokenizer)
        train_data = CachedQueryDataset(
            select=self.training_select,
            keys=self.training_keys,
            fold='train',
            transform=tokenizing_function,
        )
        if not valid_data:
            valid_data = CachedQueryDataset(
                select=self.training_select,
                # TODO: get validation keys
                keys=self.training_keys,
                fold='valid',
                transform=tokenizing_function,
            )
        else:
            if len(valid_data) > 1:
                raise TypeError('Validation sets more than one not supported yet!')
            validation_set = db.load('dataset', valid_data[0])
            valid_data = CachedQueryDataset(
                select=validation_set.select,
                # TODO: get validation keys
                keys=self.training_keys,
                fold='valid',
                transform=tokenizing_function,
            )

        return train_data, valid_data

    def _fit(  # type: ignore[override]
        self,
        X: str,
        y: str,
        select: t.Optional[Select] = None,
        db: t.Optional[BaseDatabase] = None,
        configuration: t.Optional[_TrainingConfiguration] = None,
        validation_sets: t.Optional[t.List[str]] = None,
        metrics: t.Optional[t.List[Metric]] = None,
        **kwargs,
    ):
        if configuration is not None:
            self.configuration = configuration
        if select is not None:
            self.training_select = select
        if validation_sets is not None:
            self.validation_sets = validation_sets
        if metrics is not None:
            self.metrics = metrics

        if isinstance(X, str):
            train_data, valid_data = self._get_data(db, validation_sets)

        trainer = TrainerWithSaving(
            model=self.pipeline,
            args=self.training_arguments,
            train_dataset=train_data,
            eval_dataset=valid_data,
            **kwargs,
        )
        trainer.train()

        evaluation = trainer.evalute()
        return evaluation

    def _predict(self, input, **kwargs):
        return self.object(input, **kwargs)


class TokenizingFunction:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def __call__(self, sentence):
        # TODO: discuss on this.
        # return self.tokenizer(sentence, batch=False, **self.kwargs)
        return self.tokenizer(sentence, **self.kwargs)


class TrainerWithSaving(Trainer):
    def __init__(self, custom_saver=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_saver = custom_saver

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics=metrics)
        self.custom_saver(self.args.output_dir, self.model)
