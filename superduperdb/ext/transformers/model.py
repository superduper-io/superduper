import dataclasses as dc
import typing as t

import torch
import transformers
from datasets import Dataset as NativeDataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline as BasePipeline,
    Trainer as NativeTrainer,
    TrainingArguments,
    pipeline,
    pipeline as _pipeline,
)
from transformers.pipelines.text_generation import ReturnType

from superduperdb import logging
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.base.datalayer import Datalayer
from superduperdb.components.component import ensure_initialized
from superduperdb.components.datatype import (
    DataType,
    dill_serializer,
    pickle_serializer,
)
from superduperdb.components.model import (
    Model,
    Signature,
    Trainer,
    _DeviceManaged,
    _Fittable,
    _Validator,
)
from superduperdb.ext.llm.model import BaseLLM
from superduperdb.ext.transformers.training import Checkpoint


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
class TextClassificationPipeline(Model, _Fittable, _DeviceManaged):
    """
    A wrapper for ``transformers.Pipeline``

    >>> model = TextClassificationPipeline(...)  # 123456
    >>> ,sd.s,d.s,ds
    >>>
    """

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


@dc.dataclass(kw_only=True)
class LLM(BaseLLM, _Fittable, _Validator):
    """
    LLM model based on `transformers` library.

    :param identifier: model identifier
    :param model_name_or_path: model name or path
    :param bits: quantization bits, [4, 8], default is None
    :param adapter_id: adapter id, default is None
        Add a adapter to the base model for inference.
        When model_name_or_path, bits, model_kwargs, tokenizer_kwargs are the same,
        will share the same base model and tokenizer cache.
    :param model_kwargs: model kwargs,
        all the kwargs will pass to `transformers.AutoModelForCausalLM.from_pretrained`
    :param tokenizer_kwagrs: tokenizer kwargs,
        all the kwargs will pass to `transformers.AutoTokenizer.from_pretrained`
    :param prompt_template: prompt template, default is "{input}"
    :param prompt_func: prompt function, default is None
    """

    identifier: str = ""
    model_name_or_path: t.Optional[str] = None
    adapter_id: t.Optional[t.Union[str, Checkpoint]] = None
    object: t.Optional[transformers.Trainer] = None
    model_kwargs: t.Dict = dc.field(default_factory=dict)
    tokenizer_kwargs: t.Dict = dc.field(default_factory=dict)
    prompt_template: str = "{input}"
    prompt_func: t.Optional[t.Callable] = None
    signature: str = 'singleton'
    training_kwargs: t.Dict = dc.field(default_factory=dict)

    # Save models and tokenizers cache for sharing when using multiple models
    _model_cache: t.ClassVar[dict] = {}
    _tokenizer_cache: t.ClassVar[dict] = {}

    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, DataType]]] = (
        ("model_kwargs", dill_serializer),
        ("tokenizer_kwargs", dill_serializer),
    )

    def __post_init__(self, artifacts):
        if not self.identifier:
            self.identifier = self.adapter_id or self.model_name_or_path

        #  TODO: Compatible with the bug of artifact sha1 equality and will be deleted
        self.tokenizer_kwargs.setdefault(
            "pretrained_model_name_or_path", self.model_name_or_path
        )

        super().__post_init__(artifacts)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        identifier="",
        prompt_template="{input}",
        prompt_func=None,
        **kwargs,
    ):
        """
        A new function to create a LLM model from from_pretrained function.
        Allow the user to directly replace:
        AutoModelForCausalLM.from_pretrained -> LLM.from_pretrained
        """
        model_kwargs = kwargs.copy()
        tokenizer_kwargs = {}
        return cls(
            model_name_or_path=model_name_or_path,
            identifier=identifier,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            prompt_template=prompt_template,
            prompt_func=prompt_func,
        )

    def init_pipeline(
        self, adapter_id: t.Optional[str] = None, load_adapter_directly: bool = False
    ):
        # Do not update model state here
        model_kwargs = self.model_kwargs.copy()
        tokenizer_kwargs = self.tokenizer_kwargs.copy()

        if self.model_name_or_path and not load_adapter_directly:
            model_kwargs["pretrained_model_name_or_path"] = self.model_name_or_path
            model = AutoModelForCausalLM.from_pretrained(
                **model_kwargs,
            )

            if adapter_id is not None:
                logging.info(f"Loading adapter from {adapter_id}")

                from peft import PeftModel

                try:
                    model = PeftModel.from_pretrained(
                        model, adapter_id, adapter_name=self.identifier
                    )
                except Exception as e:
                    message = (
                        f'Failed to add adapter to model, error: {e}\n'
                        'Try to load adapter directly\n'
                    )
                    logging.warn(message)
                    logging.warn("Try to load adapter directly")
                    return self.init_pipeline(adapter_id, load_adapter_directly=True)

                tokenizer_kwargs["pretrained_model_name_or_path"] = adapter_id

            else:
                tokenizer_kwargs["pretrained_model_name_or_path"] = (
                    self.model_name_or_path
                )

            tokenizer = AutoTokenizer.from_pretrained(
                **tokenizer_kwargs,
            )
        elif adapter_id is not None:
            model_kwargs['pretrained_model_name_or_path'] = adapter_id
            from peft import AutoPeftModelForCausalLM

            model = AutoPeftModelForCausalLM.from_pretrained(
                **model_kwargs,
            )
            tokenizer_kwargs["pretrained_model_name_or_path"] = adapter_id
            tokenizer = AutoTokenizer.from_pretrained(
                **tokenizer_kwargs,
            )

        else:
            raise ValueError(
                "model_name_or_path or adapter_id must be provided, "
                f"got {self.model_name_or_path} and {adapter_id} instead."
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    def init(self):
        db = self.db

        real_adapter_id = None
        if self.adapter_id is not None:
            self.handle_chekpoint(db)
            if isinstance(self.adapter_id, Checkpoint):
                checkpoint = self.adapter_id
                self.adapter_id = checkpoint.uri

            elif isinstance(self.adapter_id, str):
                real_adapter_id = self.adapter_id
                checkpoint = None
            else:
                raise ValueError(
                    "adapter_id must be either a string or Checkpoint object, but got "
                    f"{type(self.adapter_id)}"
                )

            if checkpoint:
                db = db or checkpoint.db
                assert db, "db must be provided when using checkpoint indetiifer"
                if self.db is None:
                    self.db = db
                real_adapter_id = checkpoint.path.unpack(db)

        super().init()

        self.pipeline = self.init_pipeline(real_adapter_id)

    def handle_chekpoint(self, db):
        if isinstance(self.adapter_id, str):
            # match checkpoint://<identifier>/<version>
            if Checkpoint.check_uri(self.adapter_id):
                assert db, "db must be provided when using checkpoint indetiifer"
                identifier, version = Checkpoint.parse_uri(self.adapter_id)
                version = int(version)
                checkpoint = db.load("checkpoint", identifier, version=version)
                assert checkpoint, f"Checkpoint {self.adapter_id} not found"
                self.adapter_id = checkpoint

    @ensure_initialized
    def predict_one(self, X, **kwargs):
        X = self._process_inputs(X, **kwargs)
        kwargs.pop("context", None)
        results = self._batch_generate([X], **kwargs)
        return results[0]

    @ensure_initialized
    def predict(self, dataset: t.Union[t.List, QueryDataset], **kwargs) -> t.List:
        dataset = [
            self._process_inputs(dataset[i], **kwargs) for i in range(len(dataset))
        ]
        kwargs.pop("context", None)
        return self._batch_generate(dataset, **kwargs)

    def _process_inputs(self, X: t.Any, **kwargs) -> str:
        if isinstance(X, str):
            X = self.prompter(X, **kwargs)
        return X

    def _batch_generate(self, prompts: t.List[str], **kwargs) -> t.List[str]:
        """
        Generate text.
        Can overwrite this method to support more inference methods.
        """
        kwargs = kwargs.copy()

        # Set default values, if not will cause bad output
        kwargs.setdefault("add_special_tokens", True)
        outputs = self.pipeline(
            prompts,
            return_type=ReturnType.NEW_TEXT,
            eos_token_id=self.pipeline.tokenizer.eos_token_id,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            **kwargs,
        )
        results = [output[0]["generated_text"] for output in outputs]
        return results

    def add_adapter(self, model_id, adapter_name: str):
        # TODO: Support lora checkpoint from s3
        try:
            from peft import PeftModel
        except Exception as e:
            raise ImportError("Please install peft to use LoRA training") from e

        logging.info(f"Loading adapter {adapter_name} from {model_id}")

        if not hasattr(self, "model"):
            self.init()

        if not isinstance(self.model, PeftModel):
            self.model = PeftModel.from_pretrained(
                self.model, model_id, adapter_name=adapter_name
            )
            # Update cache model
            self._model_cache[hash(self.model_kwargs)] = self.model
        else:
            # TODO where does this come from?
            self.model.load_adapter(model_id, adapter_name)

    def post_create(self, db: "Datalayer") -> None:
        # TODO: Do not make sense to add this logic here,
        # Need a auto DataType to handle this
        from superduperdb.backends.ibis.data_backend import IbisDataBackend
        from superduperdb.backends.ibis.field_types import dtype

        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype("str")
        super().post_create(db)
