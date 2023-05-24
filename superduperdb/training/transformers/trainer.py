from superduperdb.datalayer.base.query import Select
from superduperdb.models.transformers.wrapper import TokenizingFunction
from superduperdb.training.base.config import TrainerConfiguration
from transformers import Trainer, TrainingArguments


class TransformersTrainerConfiguration(TrainerConfiguration):
    def __init__(self, training_arguments: TrainingArguments, **kwargs):
        super().__init__(training_arguments=training_arguments, **kwargs)

    def __call__(
        self,
        identifier,
        models,
        keys,
        model_names,
        database_type,
        database_name,
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
            database_type,
            database_name,
            select,
            keys,
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


class TrainerWithSaving(Trainer):
    def __init__(self, custom_saver=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_saver = custom_saver

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics=metrics)
        self.custom_saver(self.args.output_dir, self.model)
