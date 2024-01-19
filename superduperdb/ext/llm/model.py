import dataclasses as dc
import functools
import os
import typing
import typing as t

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)

from superduperdb import logging
from superduperdb.backends.query_dataset import query_dataset_factory
from superduperdb.base.artifact import Artifact
from superduperdb.components.dataset import Dataset as _Dataset
from superduperdb.components.model import (
    Model,
    _TrainingConfiguration,
)
from superduperdb.ext.llm.utils import Prompter
from superduperdb.ext.utils import ensure_initialized

if typing.TYPE_CHECKING:
    from superduperdb.backends.base.query import Select
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.metric import Metric


DEFAULT_FETCH_SIZE = 10000


class LLMCallback(TrainerCallback):
    def __init__(self, llm: "LLM"):
        self.llm = llm

    def on_evaluate(self, args, state, control, **kwargs):
        self.llm.append_metrics(state.log_history[-1])


@dc.dataclass
class LLMTrainingArguments(TrainingArguments):
    """
    LLM Training Arguments.
    Inherits from :class:`transformers.TrainingArguments`.

    {training_arguments_doc}
        lora_r (`int`, *optional*, defaults to 8):
            Lora R dimension.

        lora_alpha (`int`, *optional*, defaults to 16):
            Lora alpha.

        lora_dropout (`float`, *optional*, defaults to 0.05):
            Lora dropout.

        lora_target_modules (`List[str]`, *optional*, defaults to None):
            Lora target modules. If None, will be automatically inferred.

        lora_weight_path (`str`, *optional*, defaults to ""):
            Lora weight path.

        lora_bias (`str`, *optional*, defaults to "none"):
            Lora bias.

        max_length (`int`, *optional*, defaults to 512):
            Maximum source sequence length during training.

    """

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: t.Optional[t.List[str]] = None
    lora_weight_path: str = ""
    lora_bias: str = "none"
    max_length: t.Optional[int] = 512

    __doc__ = __doc__.format(training_arguments_doc=TrainingArguments.__doc__)


@functools.wraps(LLMTrainingArguments)
def LLMTrainingConfiguration(identifier: str, **kwargs) -> _TrainingConfiguration:
    return _TrainingConfiguration(identifier=identifier, kwargs=kwargs)


@dc.dataclass
class LLM(Model):
    """
    LLM model based on `transformers` library.
    Parameters:
    : param identifier: model identifier
    : param model_name_or_path: model name or path
    : param bits: quantization bits, [4, 8], default is None
    : param adapter_id: adapter id, default is None
        Add a adapter to the base model for inference.
        When model_name_or_path, bits, model_kwargs, tokenizer_kwags are the same,
        will share the same base model and tokenizer cache.
    : param model_kwargs: model kwargs,
        all the kwargs will pass to `transformers.AutoModelForCausalLM.from_pretrained`
    : param tokenizer_kwags: tokenizer kwargs,
        all the kwargs will pass to `transformers.AutoTokenizer.from_pretrained`
    : param prompt_template: prompt template, default is "{input}"
    : param prompt_func: prompt function, default is None
    """

    identifier: str = ""
    model_name_or_path: str = "facebook/opt-125m"
    bits: t.Optional[int] = None
    adapter_id: t.Optional[str] = None
    object: t.Optional[transformers.Trainer] = None
    model_kwargs: t.Union[Artifact, t.Dict] = dc.field(default_factory=dict)
    tokenizer_kwags: t.Union[Artifact, t.Dict] = dc.field(default_factory=dict)
    prompt_template: str = "{input}"
    prompt_func: t.Optional[t.Union[Artifact, t.Callable]] = dc.field(default=None)

    # Save models and tokenizers cache for sharing when using multiple models
    _model_cache: t.ClassVar[dict] = {}
    _tokenizer_cache: t.ClassVar[dict] = {}

    def __post_init__(self):
        if not self.identifier:
            self.identifier = self.adapter_id or self.model_name_or_path

        if not isinstance(self.model_kwargs, Artifact):
            self.model_kwargs = Artifact(artifact=self.model_kwargs)
        if not isinstance(self.tokenizer_kwags, Artifact):
            self.tokenizer_kwags = Artifact(artifact=self.tokenizer_kwags)

        if not isinstance(self.prompt_func, Artifact) and self.prompt_func is not None:
            self.prompt_func = Artifact(artifact=self.prompt_func)

        # overwrite model kwargs
        if self.bits is not None:
            if (
                "load_in_4bit" in self.model_kwargs.artifact
                or "load_in_8bit" in self.model_kwargs.artifact
            ):
                logging.warn(
                    "The bits is set, will overwrite the load_in_4bit and load_in_8bit"
                )
            self.model_kwargs.artifact["load_in_4bit"] = self.bits == 4
            self.model_kwargs.artifact["load_in_8bit"] = self.bits == 8
        super().__post_init__()

    def init_model_and_tokenizer(self):
        model_key = hash(self.model_kwargs)
        if model_key not in self._model_cache:
            logging.info(f"Loading model from {self.model_name_or_path}")
            logging.info(f"model_kwargs: {self.model_kwargs.artifact}")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                **self.model_kwargs.artifact,
            )
            self._model_cache[model_key] = model
        else:
            logging.info("Reuse model from cache")

        tokenizer_key = hash(self.tokenizer_kwags)
        if tokenizer_key not in self._tokenizer_cache:
            logging.info(f"Loading tokenizer from {self.model_name_or_path}")
            logging.info(f"tokenizer_kwargs: {self.tokenizer_kwags.artifact}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                **self.tokenizer_kwags.artifact,
            )
            self._tokenizer_cache[tokenizer_key] = tokenizer
        else:
            logging.info("Reuse tokenizer from cache")
        return self._model_cache[model_key], self._tokenizer_cache[tokenizer_key]

    def create_trainer(
        self, train_dataset, eval_dataset, training_args, **kwargs
    ) -> transformers.Trainer:
        trainer = transformers.Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )
        return trainer

    def init(self):
        if self.prompt_func is not None:
            prompt_func = self.prompt_func.artifact
        else:
            prompt_func = None
        self.prompter = Prompter(self.prompt_template, prompt_func)
        self.model, self.tokenizer = self.init_model_and_tokenizer()
        if self.adapter_id is not None:
            self.add_adapter(self.adapter_id, self.adapter_id)

    def _fit(
        self,
        X: t.Any,
        y: t.Optional[t.Any] = None,
        configuration: t.Optional[_TrainingConfiguration] = None,
        data_prefetch: bool = False,
        db: t.Optional["Datalayer"] = None,
        metrics: t.Optional[t.Sequence["Metric"]] = None,
        select: t.Optional["Select"] = None,
        validation_sets: t.Optional[t.Sequence[t.Union[str, "_Dataset"]]] = None,
        **kwargs,
    ):
        assert configuration is not None, "configuration must be provided"

        training_args = LLMTrainingArguments(**configuration.kwargs)  # type: ignore

        # get device map
        device_map: t.Union[None, str, t.Dict[str, int]] = None
        if os.environ.get("LOCAL_RANK") is not None:
            device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        elif torch.backends.mps.is_available():
            device_map = "mps"

        quantization_config = self._create_quantization_config(training_args)

        logging.info("Overwriting model_kwargs for LLM training")
        logging.info(f"quantization_config: {quantization_config}")
        logging.info(f"device_map: {device_map}")

        assert isinstance(self.model_kwargs, Artifact)
        self.model_kwargs.artifact["quantization_config"] = quantization_config
        self.model_kwargs.artifact["device_map"] = device_map
        self.model, self.tokenizer = self.init_model_and_tokenizer()

        self.tokenizer.model_max_length = (
            training_args.max_length or self.tokenizer.model_max_length
        )
        self._prepare_lora_training(training_args)

        train_dataset, eval_datasets = self.get_datasets(
            X,
            y,
            db,
            select,
            db_validation_sets=validation_sets,
            data_prefetch=data_prefetch,
            prefetch_size=kwargs.pop("prefetch_size", DEFAULT_FETCH_SIZE),
        )

        # TODO: Defind callbacks about superduperdb side
        trainer = self.create_trainer(
            train_dataset,
            eval_datasets,
            compute_metrics=self.get_compute_metrics(metrics),
            training_args=training_args,
            **kwargs,
        )
        trainer.add_callback(LLMCallback(self))
        trainer.model.config.use_cache = False
        trainer.train()
        trainer.save_state()

    def get_compute_metrics(self, metrics):
        if not metrics:
            return None

        def compute_metrics(eval_preds):
            output = {}
            logits, labels = eval_preds
            for metric in metrics:
                output[metric.identifier] = metric(logits, labels)
            return output

        return compute_metrics

    @ensure_initialized
    def _predict(
        self,
        X: t.Union[str, t.List[str], t.List[dict[str, str]]],
        one: bool = False,
        **kwargs: t.Any,
    ):
        # support string and dialog format
        one = isinstance(X, str)
        if not one and isinstance(X, list):
            one = isinstance(X[0], dict)

        xs = [X] if one else X
        xs = [self.prompter(x, **kwargs) for x in xs]
        results = self._generate(xs, **kwargs)
        return results[0] if one else results

    def _generate(self, X: t.Any, adapter_name=None, **kwargs):
        """
        Private method for `Model.to_call` method.
        Support inference by multi-lora adapters.
        """
        adapter_name = adapter_name or self.adapter_id
        if adapter_name is not None:
            try:
                self.model.set_adapter(adapter_name)
                logging.info(f"Using adapter {adapter_name} for inference")
            except Exception as e:
                raise ValueError(
                    f"Adapter {adapter_name} is not found in the model, "
                    "please use add_adapter to add it."
                ) from e

        elif hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                return self._base_generate(X, **kwargs)

        return self._base_generate(X, **kwargs)

    def _base_generate(self, X: t.Any, **kwargs):
        """
        Generate text.
        Can overwrite this method to support more inference methods.
        """
        model_inputs = self.tokenizer(X, return_tensors="pt", padding=True).to(
            self.model.device
        )
        kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)
        outputs = self.model.generate(**model_inputs, **kwargs)
        texts = self.tokenizer.batch_decode(outputs)
        texts = [text.replace(self.tokenizer.eos_token, "") for text in texts]
        texts = [text.replace(self.tokenizer.pad_token, "") for text in texts]
        if isinstance(X, str):
            return texts[0]
        return texts

    def add_adapter(self, model_id, adapter_name: str):
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
            self.model.load_adapter(model_id, adapter_name)

    def _create_quantization_config(self, config: LLMTrainingArguments):
        compute_dtype = (
            torch.float16
            if config.fp16
            else (torch.bfloat16 if config.bf16 else torch.float32)
        )
        if self.bits is not None:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.bits == 4,
                load_in_8bit=self.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None
        return quantization_config

    def _prepare_lora_training(self, config: LLMTrainingArguments):
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except Exception as e:
            raise ImportError("Please install peft to use LoRA training") from e

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules
            or self._get_lora_target_modules(),
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            task_type="CAUSAL_LM",
        )

        if self.bits:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=config.gradient_checkpointing,
            )

            if not self.ddp and torch.cuda.device_count() > 1:
                self.model.is_parallelizable = True
                self.model.model_parallel = True

        self.model = get_peft_model(self.model, lora_config)

        if config.gradient_checkpointing:
            self.model.enable_input_require_grads()

        if config.local_rank == 0:
            self.model.print_trainable_parameters()

    def _get_lora_target_modules(self):
        try:
            import bitsandbytes as bnb
        except Exception as e:
            raise ImportError("Please install bitsandbytes to use LoRA training") from e

        if self.bits == 4:
            cls = bnb.nn.Linear4bit
        elif self.bits == 8:
            cls = bnb.nn.Linear8bitLt
        else:
            cls = torch.nn.Linear

        lora_module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        lora_module_names.discard("lm_head")
        return list(lora_module_names)

    def get_datasets(
        self,
        X,
        y,
        db,
        select,
        db_validation_sets: t.Optional[t.Sequence[t.Union[str, "_Dataset"]]] = None,
        data_prefetch: bool = False,
        prefetch_size: int = 10000,
    ):
        keys = [X]
        if y is not None:
            keys.append(y)

        train_dataset = query_dataset_factory(
            keys=keys,
            data_prefetch=data_prefetch,
            select=select,
            fold="train",
            db=db,
            transform=self.preprocess,
            prefetch_size=prefetch_size,
        )
        from datasets import Dataset

        train_dataset = Dataset.from_list(list(train_dataset))

        validation_sets = {}
        if db_validation_sets is None:
            logging.warn("No validation sets provided, using validation set from db")
            eval_dataset = query_dataset_factory(
                keys=keys,
                data_prefetch=data_prefetch,
                select=select,
                fold="valid",
                db=db,
                transform=self.preprocess,
                prefetch_size=prefetch_size,
            )
            eval_dataset = Dataset.from_list(list(eval_dataset))
            validation_sets["_DEFAULT"] = eval_dataset
        else:
            for _, db_dataset in enumerate(db_validation_sets):
                if isinstance(db_dataset, str):
                    db_dataset = db.load("dataset", db_dataset)

                assert isinstance(db_dataset, _Dataset), (
                    "Validation set must be a dataset, "
                    f"got {type(db_dataset)} instead."
                )

                datas = []
                for data in db_dataset.data:
                    data = data.unpack()
                    datas.append({key: data[key] for key in keys})
                dataset = Dataset.from_list(datas)
                validation_sets[db_dataset.identifier] = dataset

        def process_func(example):
            return self.tokenize(example, X, y)

        train_dataset = train_dataset.map(process_func)
        for key, dataset in validation_sets.items():
            validation_sets[key] = dataset.map(process_func)

        # If no validation sets provided, use the validation set from db
        validation_sets = validation_sets.get("_DEFAULT", validation_sets)
        return train_dataset, validation_sets

    def tokenize(self, example, X, y):
        prompt = example[X]

        prompt = prompt + self.tokenizer.eos_token
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    @property
    def ddp(self):
        return int(os.environ.get("WORLD_SIZE", 1)) != 1

    def post_create(self, db: "Datalayer") -> None:
        # TODO: Do not make sense to add this logic here,
        # Need a auto DataType to handle this
        from superduperdb.backends.ibis.data_backend import IbisDataBackend
        from superduperdb.backends.ibis.field_types import dtype

        if isinstance(db.databackend, IbisDataBackend) and self.encoder is None:
            self.encoder = dtype("str")

        super().post_create(db)
