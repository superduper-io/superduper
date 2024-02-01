import dataclasses as dc
import functools
import os
import typing
import typing as t

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from superduperdb import logging
from superduperdb.backends.query_dataset import query_dataset_factory
from superduperdb.base.artifact import Artifact
from superduperdb.components.dataset import Dataset as _Dataset
from superduperdb.components.model import (
    Model,
    _TrainingConfiguration,
)
from superduperdb.ext.llm import training
from superduperdb.ext.llm.utils import Prompter
from superduperdb.ext.utils import ensure_initialized

if typing.TYPE_CHECKING:
    from superduperdb.backends.base.query import Select
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.metric import Metric


DEFAULT_FETCH_SIZE = 10000


@functools.wraps(training.LLMTrainingArguments)
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
    adapter_id: t.Optional[t.Union[Artifact, str]] = None
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

        if self.adapter_id is not None and not isinstance(self.adapter_id, Artifact):
            self.adapter_id = Artifact(artifact=self.adapter_id)

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
            self.model_kwargs.artifact.setdefault(
                "pretrained_model_name_or_path", self.model_name_or_path
            )
            model = AutoModelForCausalLM.from_pretrained(
                **self.model_kwargs.artifact,
            )
            self._model_cache[model_key] = model
        else:
            logging.info("Reuse model from cache")

        tokenizer_key = hash(self.tokenizer_kwags)
        if tokenizer_key not in self._tokenizer_cache:
            logging.info(f"Loading tokenizer from {self.model_name_or_path}")
            logging.info(f"tokenizer_kwargs: {self.tokenizer_kwags.artifact}")
            self.tokenizer_kwags.artifact.setdefault(
                "pretrained_model_name_or_path", self.model_name_or_path
            )
            tokenizer = AutoTokenizer.from_pretrained(
                **self.tokenizer_kwags.artifact,
            )
            self._tokenizer_cache[tokenizer_key] = tokenizer

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            logging.info("Reuse tokenizer from cache")
        return self._model_cache[model_key], self._tokenizer_cache[tokenizer_key]

    def init(self):
        if self.prompt_func is not None:
            prompt_func = self.prompt_func.artifact
        else:
            prompt_func = None
        self.prompter = Prompter(self.prompt_template, prompt_func)
        self.model, self.tokenizer = self.init_model_and_tokenizer()
        if self.adapter_id is not None:
            self.add_adapter(self.adapter_id.artifact, self.adapter_id.artifact)

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

        training_config = configuration.kwargs or {}
        training_config["bits"] = training_config.get("bits", self.bits)

        train_dataset, eval_datasets = self.get_datasets(
            X,
            y,
            db,
            select,
            db_validation_sets=validation_sets,
            data_prefetch=data_prefetch,
            prefetch_size=kwargs.pop("prefetch_size", DEFAULT_FETCH_SIZE),
        )

        assert isinstance(self.model_kwargs, Artifact)
        assert isinstance(self.tokenizer_kwags, Artifact)
        model_kwargs = self.model_kwargs.artifact
        tokenizer_kwargs = self.tokenizer_kwags.artifact
        model_kwargs["pretrained_model_name_or_path"] = self.model_name_or_path
        tokenizer_kwargs["pretrained_model_name_or_path"] = self.model_name_or_path

        results = training.train(
            training_config=training_config,
            train_dataset=train_dataset,
            eval_datasets=eval_datasets,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            compute_metrics=self.get_compute_metrics(metrics),
            X=X,
            y=y,
            db=db,
            llm=self,
            **kwargs,
        )
        return results

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
        if adapter_name is None and self.adapter_id is not None:
            assert isinstance(self.adapter_id, Artifact)
            adapter_name = self.adapter_id.artifact
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
        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if isinstance(X, str):
            return texts[0]
        return texts

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
            self.model.load_adapter(model_id, adapter_name)

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

        # If no validation sets provided, use the validation set from db
        validation_sets = validation_sets.get("_DEFAULT", validation_sets)
        return train_dataset, validation_sets

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
