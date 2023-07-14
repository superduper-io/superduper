import dataclasses as dc
import functools
import typing as t

import torch
from transformers import TrainingArguments, Trainer


from superduperdb import log
from superduperdb.core.metric import Metric
from superduperdb.core.model import Model
from superduperdb.core.artifact import Artifact
from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.datalayer.base.query import Select
from superduperdb.core.model import _TrainingConfiguration
from superduperdb.datalayer.query_dataset import query_dataset_factory

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
    tokenizer: t.Optional[t.Callable] = None

    def __post_init__(self, db):
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.object.to(self.device)
        if not isinstance(self.tokenizer, Artifact):
            self.tokenizer = Artifact(_artifact=self.tokenizer)
        super().__post_init__(db)

    @property
    def pipeline(self):
        return self.object.artifact

    @functools.cached_property
    def training_arguments(self):
        return TrainingArguments(**self.configuration.kwargs)

    def _get_data(
        self,
        db,
        valid_data=None,
        data_prefetch: bool = False,
        prefetch_size: int = 100,
        X_key: str = '',
        **tokenizer_kwargs,
    ):
        tokenizing_function = TokenizingFunction(
            self.tokenizer.a, key=X_key, **tokenizer_kwargs
        )
        train_data = query_dataset_factory(
            select=self.training_select,
            keys=self.training_keys,
            fold='train',
            transform=tokenizing_function,
            data_prefetch=data_prefetch,
            prefetch_size=prefetch_size,
        )
        if not valid_data:
            valid_data = query_dataset_factory(
                select=self.training_select,
                keys=self.training_keys,
                fold='valid',
                transform=tokenizing_function,
                data_prefetch=data_prefetch,
                prefetch_size=prefetch_size,
            )
        else:
            if len(valid_data) > 1:
                raise TypeError('Validation sets more than one not supported yet!')
            validation_set = db.load('dataset', valid_data[0])
            valid_data = query_dataset_factory(
                select=validation_set.select,  # TODO: get validation keys
                keys=self.training_keys,
                fold='valid',
                transform=tokenizing_function,
                data_prefetch=data_prefetch,
                prefetch_size=prefetch_size,
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
        data_prefetch: bool = False,
        prefetch_size: int = _DEFAULT_PREFETCH_SIZE,
        tokenizer_kwargs: t.Dict[str, t.Any] = {},
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
            train_data, valid_data = self._get_data(
                db,
                validation_sets,
                data_prefetch=data_prefetch,
                prefetch_size=prefetch_size,
                X_key=X,
                **tokenizer_kwargs,
            )

        trainer = TrainerWithSaving(
            model=self.pipeline,
            args=self.training_arguments,
            train_dataset=train_data,
            eval_dataset=valid_data,
            **kwargs,
        )

        try:
            trainer.train()
            evaluation = trainer.evaluate()
        except Exception as exc:
            log.error(f"Training could not finish :: {exc}")

        return evaluation

    def _predict_one(self, input: str, **kwargs):
        tokenized_input = self.tokenizer.a(input, return_tensors='pt').to(self.device)
        return self.object.a(**tokenized_input, **kwargs)

    def _predict(self, input: str, **kwargs):
        if not isinstance(input, list):
            return self._predict_one(input, **kwargs)
        raise NotImplementedError


class TokenizingFunction:
    def __init__(self, tokenizer, key: str = '', **kwargs):
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.key = key

    def __call__(self, sentence):
        text = sentence[self.key]
        sentence.update(**self.tokenizer(text, **self.kwargs))
        return sentence


class TrainerWithSaving(Trainer):
    def __init__(self, custom_saver=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_saver = custom_saver

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics=metrics)
        if self.custom_saver:
            self.custom_saver(self.args.output_dir, self.model)
