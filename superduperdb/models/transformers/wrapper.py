import typing as t

from superduperdb.core import Encoder
from superduperdb.core.model import Model
from transformers import (
    pipeline as _pipeline,
    Pipeline as TransformersPipeline,
    TrainingArguments,
    Trainer,
)

from superduperdb.datalayer.base.query import Select
from superduperdb.training.base.config import TrainerConfiguration


# TODO - replace with TrainingArguments (no need for a new class here)
class TransformersTrainerConfiguration(TrainerConfiguration):
    def __init__(self, training_arguments: TrainingArguments, **kwargs):
        super().__init__(training_arguments=training_arguments, **kwargs)

    def __call__(
        self,
        identifier,
        models,
        keys,
        model_names,
        select: Select,
        splitter=None,
        validation_sets=(),
        metrics=None,
        features=None,
        save=None,
        download=False,
    ):
        tokenizing_function = TokenizingFunction(models[0].tokenizer)
        train_data, valid_data = self._get_data(
            select=select,
            keys=keys,
            features=features,
            transform=tokenizing_function,
        )
        optimizers = self.get('optimizers')

        args = self.training_arguments
        assert args.save_total_limit == 1, "Only top model saving supported..."
        assert args.save_strategy == 'epoch', "Only 'epoch' save strategy supported..."

        TrainerWithSaving(
            model=models[0],
            args=self.training_arguments,
            train_dataset=train_data,
            eval_dataset=valid_data,
            optimizers=optimizers,
        ).train()


class Pipeline(Model):
    def __init__(
        self,
        pipeline: t.Optional[TransformersPipeline] = None,
        task: t.Optional[str] = None,
        model: t.Optional[str] = None,
        identifier: t.Optional[str] = None,
        training_configuration: t.Optional[TransformersTrainerConfiguration] = None,
        training_select: t.Optional[Select] = None,
        training_keys: t.Optional[t.Dict] = None,
        encoder: t.Optional[Encoder] = None,
    ):
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

    def predict_one(self, r, **kwargs):
        return self.object(r, **kwargs)

    def predict(self, docs, **kwargs):
        return self.object(docs, **kwargs)


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
