import dataclasses as dc
import typing as t

import torch
import transformers
from datasets import Dataset as NativeDataset
from superduper import logging
from superduper.backends.query_dataset import QueryDataset
from superduper.base.datalayer import Datalayer
from superduper.components.component import ensure_initialized
from superduper.components.model import (
    Model,
    Signature,
    Trainer,
    _DeviceManaged,
)
from superduper.components.training import Checkpoint
from superduper.ext.llm.model import BaseLLM
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


class _TrainerWithSaving(NativeTrainer):
    def __init__(self, custom_saver=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_saver = custom_saver

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics=metrics)
        logging.info('Saving model...')
        if self.custom_saver:
            self.custom_saver()


class TransformersTrainer(TrainingArguments, Trainer):
    """Trainer for transformers models # noqa.

    It's used to train the transformers models.

    :param signature: signature, default is '**kwargs'
    :param output_dir: output directory
    :param data_collator: data collator
    :param callbacks: callbacks for training
    :param optimizers: optimizers for training
    :param preprocess_logits_for_metrics: preprocess logits for metrics
    """

    _fields = {
        'data_collator': 'default',
        'callbacks': 'default',
        'optimizers': 'default',
        'preprocess_logits_for_metrics': 'default',
    }

    signature: Signature = '**kwargs'
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

    def __post_init__(self, db):
        assert self.output_dir == '' or self.output_dir == self.identifier
        self.output_dir = self.identifier
        TrainingArguments.__post_init__(self)
        return Trainer.__post_init__(self, db)

    @property
    def native_arguments(self):
        """Get native arguments of TrainingArguments."""
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

        train_dataset = [train_dataset[i] for i in range(len(train_dataset))]
        train_dataset = NativeDataset.from_list(train_dataset)
        train_dataset = train_dataset.map(preprocess_function)
        valid_dataset = [valid_dataset[i] for i in range(len(valid_dataset))]
        valid_dataset = NativeDataset.from_list(valid_dataset)
        valid_dataset = valid_dataset.map(preprocess_function)

        def compute_metrics(eval_pred):
            output = {}
            for vs in model.validation_sets:
                output[vs.identifier] = model.validate(vs)
            self.append_metrics(output)
            return output

        def custom_saver():
            db.replace(model)

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
        """Fit the model.

        :param model: model
        :param db: Datalayer instance
        :param train_dataset: training dataset
        :param valid_dataset: validation dataset
        """
        trainer = self._build_trainer(
            model=model,
            db=db,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
        )
        trainer.train()


class TextClassificationPipeline(Model, _DeviceManaged):
    """A wrapper for ``transformers.Pipeline``.

    :param tokenizer_name: tokenizer name
    :param tokenizer_cls: tokenizer class, e.g. ``transformers.AutoTokenizer``
    :param tokenizer_kwargs: tokenizer kwargs, will pass to ``tokenizer_cls``
    :param model_name: model name, will pass to ``model_cls``
    :param model_cls: model class, e.g. ``AutoModelForSequenceClassification``
    :param model_kwargs: model kwargs, will pass to ``model_cls``
    :param pipeline: pipeline instance, default is None, will build when None
    :param task: task of the pipeline
    :param trainer: `TransformersTrainer` instance
    :param preferred_devices: preferred devices
    :param device: device to use

    Example:
    -------
    >>> from superduper_transformers.model import TextClassificationPipeline
    >>>
    >>> model = TextClassificationPipeline(
    >>>     identifier="my-sentiment-analysis",
    >>>     model_name="distilbert-base-uncased",
    >>>     model_kwargs={"num_labels": 2},
    >>>     device="cpu",
    >>> )
    >>> model.predict("Hello, world!")

    """

    _fields = {
        'tokenizer_cls': 'default',
        'model_cls': 'default',
        'pipeline': 'default',
    }
    signature: Signature = 'singleton'
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

    def __post_init__(self, db, example):
        if self.pipeline is None:
            self._build_pipeline()
        super().__post_init__(db, example)

    def predict(self, text: str):
        """Predict the class of a single text.

        :param text: a text
        """
        return self.pipeline(text)[0]

    def predict_batches(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        """Predict the class of a list of text.

        :param dataset: a list of text
        """
        text = [dataset[i] for i in range(len(dataset))]
        return self.pipeline(text)


class LLM(BaseLLM):
    """
    LLM model based on `transformers` library.

    :param identifier: model identifier
    :param model_name_or_path: model name or path
    :param adapter_id: adapter id, default is None
        Add a adapter to the base model for inference.
    :param model_kwargs: model kwargs,
        all the kwargs will pass to `transformers.AutoModelForCausalLM.from_pretrained`
    :param tokenizer_kwargs: tokenizer kwargs,
        all the kwargs will pass to `transformers.AutoTokenizer.from_pretrained`
    :param prompt_template: prompt template, default is `"{input}"`
    :param prompt_func: prompt function, default is None
    :param trainer: Trainer to use to handle training details

    All the `model_kwargs` will pass to
    `transformers.AutoModelForCausalLM.from_pretrained`.
    All the `tokenize_kwargs` will pass to
    `transformers.AutoTokenizer.from_pretrained`.
    When `model_name_or_path`, `bits`, `model_kwargs`, `tokenizer_kwargs` are the same,
    will share the same base model and tokenizer cache.

    Example:
    -------
    >>> from superduper_transformers import LLM
    >>> model = LLM(identifier="llm", model_name_or_path="facebook/opt-125m")
    >>> model.predict("Hello, world!")

    """

    identifier: str = ""
    model_name_or_path: t.Optional[str] = None
    adapter_id: t.Optional[t.Union[str, Checkpoint]] = None
    model_kwargs: t.Dict = dc.field(default_factory=dict)
    tokenizer_kwargs: t.Dict = dc.field(default_factory=dict)
    prompt_template: str = "{input}"
    prompt_func: t.Optional[t.Callable] = None
    signature: Signature = 'singleton'

    # Save models and tokenizers cache for sharing when using multiple models
    _model_cache: t.ClassVar[dict] = {}
    _tokenizer_cache: t.ClassVar[dict] = {}

    _fields = {
        'model_kwargs': 'default',
        'tokenizer_kwargs': 'default',
    }

    def __post_init__(self, db, example):
        if not self.identifier:
            self.identifier = self.adapter_id or self.model_name_or_path

        #  TODO: Compatible with the bug of artifact sha1 equality and will be deleted
        super().__post_init__(db, example)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        identifier="",
        prompt_template="{input}",
        prompt_func=None,
        predict_kwargs=None,
        **kwargs,
    ):
        """A new function to create a LLM model from from_pretrained function.

        Allow the user to directly replace:
        `AutoModelForCausalLM.from_pretrained` -> `LLM.from_pretrained`

        :param model_name_or_path: model name or path
        :param identifier: model identifier
        :param prompt_template: prompt template, default is `"{input}"`
        :param prompt_func: prompt function, default is None
        :param predict_kwargs: predict kwargs, default is None
        :param kwargs: additional keyword arguments, all the kwargs will pass to `LLM`
        """
        model_kwargs = kwargs.copy()
        tokenizer_kwargs = {}
        predict_kwargs = predict_kwargs or {}
        return cls(
            model_name_or_path=model_name_or_path,
            identifier=identifier,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            prompt_template=prompt_template,
            prompt_func=prompt_func,
            predict_kwargs=predict_kwargs,
        )

    def init_pipeline(
        self, adapter_id: t.Optional[str] = None, load_adapter_directly: bool = False
    ):
        """Initialize pipeline.

        :param adapter_id: adapter id
        :param load_adapter_directly: load adapter directly
        """
        # Do not update model state here
        model_kwargs = self.model_kwargs.copy()

        tokenizer_kwargs = self.tokenizer_kwargs.copy()
        tokenizer_kwargs.setdefault(
            "pretrained_model_name_or_path", self.model_name_or_path
        )

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
                tokenizer_kwargs[
                    "pretrained_model_name_or_path"
                ] = self.model_name_or_path

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

    def init(self, db=None):
        """Initialize the model.

        If adapter_id is provided, will load the adapter to the model.
        """
        super().init()
        real_adapter_id = None
        if self.adapter_id is not None:
            if isinstance(self.adapter_id, Checkpoint):
                real_adapter_id = self.adapter_id.path

            elif isinstance(self.adapter_id, str):
                real_adapter_id = self.adapter_id

        self.pipeline = self.init_pipeline(real_adapter_id)

    @ensure_initialized
    def predict(self, X, **kwargs):
        """Generate text from a single prompt.

        :param X: a prompt
        :param kwargs: additional keyword arguments
        """
        X = self._process_inputs(X, **kwargs)
        kwargs.pop("context", None)
        results = self._batch_generate([X], **kwargs)
        return results[0]

    @ensure_initialized
    def predict_batches(
        self, dataset: t.Union[t.List, QueryDataset], **kwargs
    ) -> t.List:
        """Generate text from a list of prompts.

        :param dataset: a list of prompts
        :param kwargs: additional keyword arguments
        """
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
        """Generate text.

        Can overwrite this method to support more inference methods.
        """
        kwargs = {**self.predict_kwargs, **kwargs.copy()}

        # Set default values, if not will cause bad output
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
        """Add adapter to the model.

        :param model_id: model id
        :param adapter_name: adapter name
        """
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
