import dataclasses as dc
import functools
import typing as t

from superduperdb.core.metric import Metric
from superduperdb.core.model import Model
from transformers import (
    TrainingArguments,
    Trainer,
)

from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.datalayer.base.query import Select
from superduperdb.core.model import _TrainingConfiguration
from superduperdb.datalayer.query_dataset import QueryDataset


@functools.wraps(TrainingArguments)
def TransformersTrainerConfiguration(identifier: str, *args, **kwargs):
    cfg = TrainingArguments(output_dir=args[0], **kwargs)
    return _TrainingConfiguration(identifier=identifier, kwargs=cfg.to_dict())


@dc.dataclass
class Pipeline(Model):
    @property
    def pipeline(self):
        return self.object.a

    @functools.cached_property
    def training_arguments(self):
        return TrainingArguments(**self.training_configuration.dict())

    def _get_data(self):
        tokenizing_function = TokenizingFunction(self.object.tokenizer)
        train_data = QueryDataset(
            select=self.training_select,
            keys=self.training_keys,
            fold='train',
            transform=tokenizing_function,
        )
        valid_data = QueryDataset(
            select=self.training_select,
            keys=self.training_keys,
            fold='valid',
            transform=tokenizing_function,
        )
        train_data = [train_data[i] for i in range(len(train_data))]
        valid_data = [valid_data[i] for i in range(len(valid_data))]
        return train_data, valid_data

    def _fit(  # type: ignore[override]
        self,
        X: str,
        y: str,
        select: t.Optional[Select] = None,
        database: t.Optional[BaseDatabase] = None,
        training_configuration: t.Optional[_TrainingConfiguration] = None,
        validation_sets: t.Optional[t.List[str]] = None,
        metrics: t.Optional[t.List[Metric]] = None,
    ):
        if training_configuration is not None:
            self.training_configuration = training_configuration
        if select is not None:
            self.training_select = select
        if validation_sets is not None:
            self.validation_sets = validation_sets
        if metrics is not None:
            self.metrics = metrics

        if isinstance(X, str):
            train_data, valid_data = self._get_data()
            X_train = []
            y_train = []
            for i in range(len(train_data)):
                r = train_data[i]
                X_train.append(r[X])
                y_train.append(r[y])

        TrainerWithSaving(
            model=self.object,
            args=self.training_arguments,
            train_dataset=train_data,
            eval_dataset=valid_data,
        ).train()

    def _predict(self, input, **kwargs):
        return self.object(input, **kwargs)


class TokenizingFunction:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def __call__(self, sentence):
        return self.tokenizer(sentence, batch=False, **self.kwargs)


class TrainerWithSaving(Trainer):
    def __init__(self, custom_saver=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_saver = custom_saver

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics=metrics)
        self.custom_saver(self.args.output_dir, self.model)
