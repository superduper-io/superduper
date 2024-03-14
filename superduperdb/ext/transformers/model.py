import dataclasses as dc
import typing as t

import torch
import transformers
from datasets import Dataset as NativeDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline as BasePipeline,
    Trainer as NativeTrainer,
    TrainingArguments,
    pipeline as _pipeline,
)

from superduperdb import logging
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.base.datalayer import Datalayer
from superduperdb.components.datatype import (
    DataType,
    dill_serializer,
    pickle_serializer,
)
from superduperdb.components.model import (
    Signature,
    Trainer,
    _DeviceManaged,
    _Fittable,
    _Predictor,
)


class _TrainerWithSaving(NativeTrainer):
    def __init__(self, custom_saver=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_saver = custom_saver

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics=metrics)
        logging.info('Saving model...')
        if self.custom_saver:
            self.custom_saver()


@dc.dataclass(kw_only=True)
class TransformersTrainer(TrainingArguments, Trainer):
    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, DataType]]] = (
        ('data_collator', dill_serializer),
        ('callbacks', dill_serializer),
        ('optimizers', dill_serializer),
        ('preprocess_logits_for_metrics', dill_serializer),
    )

    output_dir: str = ''
    data_collator: t.Optional[transformers.data.data_collator.DataCollator] = None
    callbacks: t.Optional[t.List[transformers.trainer_callback.TrainerCallback]] = None
    optimizers: t.Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
        None,
        None,
    )
    preprocess_logits_for_metrics: t.Optional[
        t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None

    def __post_init__(self, artifacts):
        assert self.output_dir == '' or self.output_dir == self.identifier
        self.output_dir = self.identifier
        TrainingArguments.__post_init__(self)
        return Trainer.__post_init__(self, artifacts)

    @property
    def native_arguments(self):
        _TRAINING_DEFAULTS = {
            k: v
            for k, v in TrainingArguments('_tmp').to_dict().items()
            if k != 'output_dir'
        }
        kwargs = {k: getattr(self, k) for k in _TRAINING_DEFAULTS}
        return TrainingArguments(output_dir=self.identifier, **kwargs)

    def _build_trainer(
        self,
        model: 'TextClassificationPipeline',
        db: Datalayer,
        train_dataset: QueryDataset,
        valid_dataset: QueryDataset,
    ):
        def preprocess_function(examples):
            return model.pipeline.tokenizer(
                examples['text'], padding="max_length", truncation=True
            )

        train_dataset = NativeDataset.from_list(
            [train_dataset[i] for i in range(len(train_dataset))]
        )
        train_dataset = train_dataset.map(preprocess_function)
        valid_dataset = NativeDataset.from_list(
            [valid_dataset[i] for i in range(len(valid_dataset))]
        )
        valid_dataset = valid_dataset.map(preprocess_function)

        def compute_metrics(eval_pred):
            output = {}
            for vs in model.validation_sets:
                output[vs.identifier] = model.validate(vs)
            self.append_metrics(output)
            return output

        def custom_saver():
            db.replace(model, upsert=True)

        trainer = _TrainerWithSaving(
            model=model.pipeline.model,  # type: ignore[union-attr]
            args=self.native_arguments,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            custom_saver=custom_saver,
            compute_metrics=compute_metrics,
            data_collator=self.data_collator,
            callbacks=self.callbacks,
            optimizers=self.optimizers,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
        )
        return trainer

    def fit(
        self,
        model: 'TextClassificationPipeline',
        db: Datalayer,
        train_dataset: QueryDataset,
        valid_dataset: QueryDataset,
    ):
        trainer = self._build_trainer(
            model=model,
            db=db,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
        )
        trainer.train()


@dc.dataclass(kw_only=True)
class TextClassificationPipeline(_Predictor, _Fittable, _DeviceManaged):
    """A wrapper for ``transformers.Pipeline``"""

    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = (
        ('tokenizer_cls', pickle_serializer),
        ('model_cls', pickle_serializer),
        ('object', pickle_serializer),
    )
    signature: t.ClassVar[Signature] = 'singleton'
    train_signature: t.ClassVar[Signature] = '**kwargs'
    tokenizer_name: t.Optional[str] = None
    tokenizer_cls: object = AutoTokenizer
    tokenizer_kwargs: t.Dict = dc.field(default_factory=dict)
    model_name: t.Optional[str] = None
    model_cls: object = AutoModelForSequenceClassification
    model_kwargs: t.Dict = dc.field(default_factory=dict)
    pipeline: t.Optional[BasePipeline] = None
    task: str = 'text-classification'

    def _build_pipeline(self):
        self.pipeline = _pipeline(
            task=self.task,
            tokenizer=self.tokenizer_cls.from_pretrained(
                self.tokenizer_name or self.model_name
            ),
            model=self.model_cls.from_pretrained(self.model_name),
        )

    def __post_init__(self, artifacts):
        if self.pipeline is None:
            self._build_pipeline()
        super().__post_init__(artifacts)

    def predict_one(self, text: str):
        return self.pipeline(text)[0]

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        text = [dataset[i] for i in range(len(dataset))]
        return self.pipeline(text)
