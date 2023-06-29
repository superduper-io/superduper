import typing as t

from superduperdb.core.encoder import Encoder
from superduperdb.core.metric import Metric
from superduperdb.core.model import Model
from transformers import (
    pipeline as _pipeline,
    Pipeline as TransformersPipeline,
    TrainingArguments,
    Trainer,
)

from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.datalayer.base.query import Select
from superduperdb.core.model import TrainingConfiguration
from superduperdb.datalayer.query_dataset import QueryDataset


class TransformersTrainerConfiguration(TrainingConfiguration):
    training_arguments: TrainingArguments

    def __init__(self, training_arguments: TrainingArguments, **kwargs):
        super().__init__(training_arguments=training_arguments, **kwargs)


class Pipeline(Model):
    def __init__(
        self,
        pipeline: t.Optional[TransformersPipeline] = None,
        task: t.Optional[str] = None,
        model: t.Optional[str] = None,
        identifier: t.Optional[str] = None,
        training_configuration: t.Optional[TransformersTrainerConfiguration] = None,
        training_select: t.Optional[Select] = None,
        training_keys: t.Optional[t.List[str]] = None,
        encoder: t.Optional[Encoder] = None,
    ) -> None:
        if pipeline is None:
            assert model is not None, 'must specify model for now'
            pipeline = _pipeline(task, model=model)

        identifier = identifier or f'{pipeline.task}/{pipeline.model.name_or_path}'

        super().__init__(
            pipeline,
            identifier=identifier,
            training_configuration=training_configuration,
            training_select=training_select,
            training_keys=training_keys,
            encoder=encoder,
        )

    def _get_data(self) -> t.Tuple[QueryDataset, QueryDataset]:
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

    def fit(
        self,
        X: str,
        y: str,
        select: t.Optional[Select] = None,
        database: t.Optional[BaseDatabase] = None,
        training_configuration: t.Optional[TransformersTrainerConfiguration] = None,
        validation_sets: t.Optional[t.List[str]] = None,
        metrics: t.Optional[t.List[Metric]] = None,
        serializer: str = 'pickle',
    ) -> t.Any:
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

        # ruff: noqa: E501
        TrainerWithSaving(
            model=self.object,
            args=self.training_configuration.training_arguments,  # type: ignore[union-attr]
            train_dataset=train_data,
            eval_dataset=valid_data,
        ).train()

    def predict(self, input, **kwargs):
        return self.object(input, **kwargs)


class TokenizingFunction:
    def __init__(self, tokenizer: t.Callable, **kwargs: t.Any) -> None:
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def __call__(self, sentence) -> t.Any:
        return self.tokenizer(sentence, batch=False, **self.kwargs)


class TrainerWithSaving(Trainer):
    def __init__(self, custom_saver=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_saver = custom_saver

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics=metrics)
        self.custom_saver(self.args.output_dir, self.model)
