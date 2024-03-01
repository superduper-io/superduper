import dataclasses as dc
import functools
import typing
import typing as t

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from transformers.pipelines.text_generation import ReturnType

from superduperdb import logging
from superduperdb.backends.query_dataset import QueryDataset, query_dataset_factory
from superduperdb.components.component import ensure_initialized
from superduperdb.components.dataset import Dataset as _Dataset
from superduperdb.components.datatype import DataType, dill_serializer
from superduperdb.components.model import (
    _Fittable,
    _Predictor,
    _TrainingConfiguration,
)
from superduperdb.ext.llm import training
from superduperdb.ext.llm.utils import Prompter

from .training import Checkpoint

if typing.TYPE_CHECKING:
    from datasets import Dataset

    from superduperdb.backends.base.query import Select
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.metric import Metric


DEFAULT_FETCH_SIZE = 10000


@functools.wraps(training.LLMTrainingArguments)
def LLMTrainingConfiguration(identifier: str, **kwargs) -> _TrainingConfiguration:
    return _TrainingConfiguration(identifier=identifier, kwargs=kwargs)


@dc.dataclass
class LLM(_Predictor, _Fittable):
    """
    LLM model based on `transformers` library.
    Parameters:
    : param identifier: model identifier
    : param model_name_or_path: model name or path
    : param bits: quantization bits, [4, 8], default is None
    : param adapter_id: adapter id, default is None
        Add a adapter to the base model for inference.
        When model_name_or_path, bits, model_kwargs, tokenizer_kwargs are the same,
        will share the same base model and tokenizer cache.
    : param model_kwargs: model kwargs,
        all the kwargs will pass to `transformers.AutoModelForCausalLM.from_pretrained`
    : param tokenizer_kwagrs: tokenizer kwargs,
        all the kwargs will pass to `transformers.AutoTokenizer.from_pretrained`
    : param prompt_template: prompt template, default is "{input}"
    : param prompt_func: prompt function, default is None
    """

    _encodables: t.ClassVar[t.Sequence[str]] = ("model_kwargs", "tokenizer_kwargs")

    identifier: str = ""
    model_name_or_path: t.Optional[str] = None
    adapter_id: t.Optional[t.Union[str, Checkpoint]] = None
    object: t.Optional[transformers.Trainer] = None
    model_kwargs: t.Dict = dc.field(default_factory=dict)
    tokenizer_kwargs: t.Dict = dc.field(default_factory=dict)
    prompt_template: str = "{input}"
    prompt_func: t.Optional[t.Callable] = None
    signature: str = "singleton"  # type: ignore[misc]

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
        self.prompter = Prompter(self.prompt_template, self.prompt_func)

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
        train_dataset: t.Optional["Dataset"] = None,
        eval_dataset: t.Optional[t.Union["Dataset", t.Dict[str, "Dataset"]]] = None,
        **kwargs,
    ):
        assert configuration is not None, "configuration must be provided"

        training_config = configuration.kwargs or {}

        if not train_dataset:
            train_dataset, eval_dataset = self.get_datasets(
                X,
                y,
                db,
                select,
                db_validation_sets=validation_sets,
                data_prefetch=data_prefetch,
                prefetch_size=kwargs.pop("prefetch_size", DEFAULT_FETCH_SIZE),
            )

        assert train_dataset is not None, "train_dataset must be provided"

        model_kwargs = self.model_kwargs.copy()
        tokenizer_kwargs = self.tokenizer_kwargs.copy()
        assert (
            self.model_name_or_path
        ), "model_name_or_path must be provided for training"
        model_kwargs["pretrained_model_name_or_path"] = self.model_name_or_path
        tokenizer_kwargs["pretrained_model_name_or_path"] = self.model_name_or_path

        results = training.train(
            training_config=training_config,
            train_dataset=train_dataset,
            eval_datasets=eval_dataset,
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
    def predict_one(self, X, **kwargs):
        X = self._process_inputs(X, **kwargs)
        results = self._generate([X], **kwargs)
        return results[0]

    @ensure_initialized
    def predict(self, dataset: t.Union[t.List, QueryDataset], **kwargs) -> t.List:
        dataset = [
            self._process_inputs(dataset[i], **kwargs) for i in range(len(dataset))
        ]
        return self._generate(dataset, **kwargs)

    def _process_inputs(self, X: t.Any, **kwargs) -> str:
        if isinstance(X, str):
            X = self.prompter(X, **kwargs)
        return X

    def _generate(self, X: list, **kwargs):
        """
        Generate text.
        Can overwrite this method to support more inference methods.
        """
        kwargs = kwargs.copy()

        # Set default values, if not will cause bad output
        kwargs.setdefault("add_special_tokens", True)
        outputs = self.pipeline(
            X,
            return_type=ReturnType.NEW_TEXT,
            eos_token_id=self.pipeline.tokenizer.eos_token_id,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            **kwargs,
        )
        results = [output[0]["generated_text"] for output in outputs]
        return results

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

        def transform(x):
            if "_id" in x:
                x["_id"] = str(x["_id"])
            return x

        train_dataset = query_dataset_factory(
            keys=keys,
            data_prefetch=data_prefetch,
            select=select,
            fold="train",
            db=db,
            transform=transform,
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
                transform=transform,
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

    def post_create(self, db: "Datalayer") -> None:
        # TODO: Do not make sense to add this logic here,
        # Need a auto DataType to handle this
        from superduperdb.backends.ibis.data_backend import IbisDataBackend
        from superduperdb.backends.ibis.field_types import dtype

        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype("str")
        super().post_create(db)
