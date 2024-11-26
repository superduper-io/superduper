import dataclasses as dc
import os
import typing as t
from copy import deepcopy
from functools import wraps

import torch
import transformers
from datasets import Dataset as NativeDataset
from superduper import logging
from superduper.backends.query_dataset import QueryDataset
from superduper.base.build import build_datalayer
from superduper.base.config import Config
from superduper.base.datalayer import Datalayer
from superduper.components.dataset import Dataset
from superduper.components.model import Trainer as SuperDuperTrainer
from superduper.components.training import Checkpoint
from superduper.misc.hash import random_sha1
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

if t.TYPE_CHECKING:
    from superduper.ext.transformers.model import LLM


class LLMCallback(TrainerCallback):
    """LLM Callback for logging training process to db # noqa.

    This callback will save the checkpoint to db after each epoch.
    If the save_total_limit is set, will remove the oldest checkpoint.

    :param cfg: The configuration to use.
    :param identifier: The identifier to use.
    :param db: The datalayer to use.
    :param llm: The LLM model to use.
    :param experiment_id: The experiment id to use.
    """

    def __init__(
        self,
        cfg: t.Optional["Config"] = None,
        identifier: t.Optional[str] = None,
        db: t.Optional["Datalayer"] = None,
        llm: t.Optional["LLM"] = None,
        experiment_id: t.Optional[str] = None,
    ):
        self.cfg = cfg
        self.identifier = identifier
        self.db = db
        self.llm = llm
        self.experiment_id = experiment_id or random_sha1()

        # If we run training on remote, we need to provide identifier and cfg,
        # then can connect to db and load llm
        is_remote_init = self.identifier is not None and self.cfg is not None

        # If we run training on local, we can provide db and llm directly
        is_local_init = self.db is not None and self.llm is not None

        if not (is_remote_init or is_local_init):
            raise ValueError(
                "Please provide either (identifier and cfg) or (db and llm)"
            )

    def on_save(self, args, state, control, **kwargs):
        """Event called after a checkpoint save.

        :param args: The training arguments from transformers.
        :param state: The training state from transformers.
        :param control: The training control from transformers.
        :param kwargs: Other keyword arguments from transformers.
        """
        if not state.is_world_process_zero:
            return

        self.check_init()
        checkpoint_path = transformers.trainer.get_last_checkpoint(args.output_dir)
        if checkpoint_path is None:
            logging.warn("No checkpoint found, skip saving checkpoint")
            return

        checkpoint = Checkpoint(
            identifier=self.experiment_id, path=checkpoint_path, step=state.global_step
        )
        self.llm.adapter_id = checkpoint
        self.db.replace(self.llm)

        if not args.save_total_limit:
            return

        try:
            versions = self.db.show("checkpoint", self.experiment_id) or []
        except Exception:
            versions = []
        if len(versions) > args.save_total_limit:
            for version in versions[: -args.save_total_limit]:
                self.db.remove("checkpoint", self.experiment_id, version, force=True)

    def on_evaluate(self, args, state, control, **kwargs):
        """Event called after an evaluation.

        :param args: The training arguments from transformers.
        :param state: The training state from transformers.
        :param control: The training control from transformers.
        :param kwargs: Other keyword arguments from transformers.
        """
        if not state.is_world_process_zero:
            return

        self.check_init()
        self.llm.append_metrics(state.log_history[-1])

    def on_train_end(self, args, state, control, **kwargs):
        """Event called after training ends.

        :param args: The training arguments from transformers.
        :param state: The training state from transformers.
        :param control: The training control from transformers.
        :param kwargs: Other keyword arguments from transformers.
        """
        self.check_init()
        # update the llm to db after training, will save the adapter_id and metrics

        if state.best_model_checkpoint:
            step = state.best_model_checkpoint.split("-")[-1]
            checkpoint = Checkpoint(
                identifier=self.experiment_id,
                path=state.best_model_checkpoint,
                step=step,
            )
            self.llm.adapter_id = checkpoint
            self.db.replace(self.llm)

    def check_init(self):
        """Check the initialization of the callback."""
        # Only check this in the world_rank 0 process
        # Rebuild datalayer for the new process
        if self.db is None:
            self.db = build_datalayer(self.cfg)
            self.llm = self.db.load("model", self.identifier)

        assert self.llm is not None


class LLMTrainer(TrainingArguments, SuperDuperTrainer):
    """LLM Training Arguments # noqa.

    Inherits from :class:`transformers.TrainingArguments`.

    :param output_dir: The output directory to use.
    :param use_lora: Whether to use LoRA training.
    :param lora_r: Lora R dimension.
    :param lora_alpha: Lora alpha.
    :param lora_dropout: Lora dropout.
    :param lora_target_modules: Lora target modules.
    :param lora_bias: Lora bias.
    :param bits: The bits to use.
    :param max_seq_length: Maximum source sequence length during training.
    :param setup_chat_format: Whether to setup chat format.
    :param log_to_db: Whether to log training to db.
    :param training_kwargs: The training kwargs to use, will be passed to Trainer.
    :param num_gpus: The number of GPUs to use, if None, will use all GPUs.
    :param ray_configs: The ray configs to use.
    """

    output_dir: str = ''
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: t.Union[t.List[str], str] = "all-linear"
    lora_bias: t.Literal["none", "all", "lora_only"] = "none"
    bits: t.Optional[int] = None
    max_seq_length: int = 512
    setup_chat_format: bool = False
    log_to_db: bool = True
    training_kwargs: t.Dict = dc.field(default_factory=dict)
    num_gpus: t.Optional[int] = None
    ray_configs: t.Optional[dict] = None

    def __post_init__(self, db):
        self.output_dir = self.output_dir or os.path.join("output", self.identifier)
        if self.num_gpus and 'num_gpus' not in self.compute_kwargs:
            self.compute_kwargs['num_gpus'] = self.num_gpus
        return SuperDuperTrainer.__post_init__(self, db)

    def build(self):
        """Build the training arguments."""
        super().__post_init__()

    def build_training_args(self, build_class=TrainingArguments):
        """Build the training arguments."""
        _TRAINING_DEFAULTS = {
            k: v
            for k, v in TrainingArguments('_tmp').to_dict().items()
            if k != 'output_dir'
        }
        kwargs = {k: getattr(self, k) for k in _TRAINING_DEFAULTS}
        return build_class(output_dir=self.output_dir, **kwargs)

    @staticmethod
    def get_compute_metrics(metrics):
        """Get the compute metrics function.

        :param metrics: List of callable metric functions.
                        Each function should take logits and labels as input
                        and return a metric value.

        """
        if not metrics:
            return None

        def compute_metrics(eval_preds):
            output = {}
            logits, labels = eval_preds
            for metric in metrics:
                output[metric.identifier] = metric(logits, labels)
            return output

        return compute_metrics

    def prepare_dataset(self, model, dataset: QueryDataset):
        """Prepare the dataset for training.

        :param model: The model to use.
        :param dataset: The dataset to prepare.
        """
        if isinstance(self.key, str):
            dataset.transform = lambda x: {self.key: x}

    def fit(
        self,
        model: 'LLM',
        db: Datalayer,
        train_dataset: t.Union[QueryDataset, NativeDataset],
        valid_dataset: t.Union[QueryDataset, NativeDataset],
    ):
        """Fit the model on the training dataset.

        :param model: The model to fit.
        :param db: The datalayer to use.
        :param train_dataset: The training dataset to use.
        :param valid_dataset: The validation dataset to use.
        """
        if isinstance(train_dataset, QueryDataset):
            self.prepare_dataset(model, train_dataset)
            train_dataset = NativeDataset.from_list(
                list(train_dataset)  # type: ignore[call-overload]
            )

        eval_datasets = {}

        if model.validation:
            for vs in model.validation.datasets:
                qvs = model._create_dataset(model.validation.key, db, select=vs.select)
                self.prepare_dataset(model, qvs)
                eval_datasets[vs.identifier] = NativeDataset.from_list(list(qvs))
        if isinstance(valid_dataset, QueryDataset):
            self.prepare_dataset(model, valid_dataset)
            valid_dataset = NativeDataset.from_list(
                list(valid_dataset)  # type: ignore[call-overload]
            )

        if eval_datasets:
            eval_datasets['_default'] = valid_dataset
        else:
            eval_datasets = valid_dataset

        model_kwargs = model.model_kwargs.copy()
        tokenizer_kwargs = model.tokenizer_kwargs.copy()
        assert (
            model.model_name_or_path
        ), "model_name_or_path must be provided for training"
        model_kwargs["pretrained_model_name_or_path"] = model.model_name_or_path
        tokenizer_kwargs["pretrained_model_name_or_path"] = model.model_name_or_path

        self._experiment_id = random_sha1()
        logging.info(f"Start training, experiment_id: {self._experiment_id}")

        return train(
            training_args=self,
            train_dataset=train_dataset,
            eval_datasets=valid_dataset,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            compute_metrics=self.get_compute_metrics(
                model.validation.metrics if model.validation else None
            ),
            db=db,
            llm=model,
            ray_configs=self.ray_configs,
            **(self.training_kwargs or {}).copy(),
        )

    @property
    def experiment_id(self):
        """Get the experiment id."""
        return getattr(self, "_experiment_id", None)


def tokenize(tokenizer, example, X, y):
    """Function to tokenize the example.

    :param tokenizer: The tokenizer to use.
    :param example: The example to tokenize.
    :param X: The input key.
    :param y: The output key.
    """
    prompt = example[X]

    prompt = prompt + tokenizer.eos_token
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


def train(
    training_args: LLMTrainer,
    train_dataset: NativeDataset,
    eval_datasets: t.Union[NativeDataset, t.Dict[str, NativeDataset]],
    model_kwargs: dict,
    tokenizer_kwargs: dict,
    db: t.Optional["Datalayer"] = None,
    llm: t.Optional["LLM"] = None,
    ray_configs: t.Optional[dict] = None,
    **kwargs,
):
    """Train LLM model on specified dataset.

    The training process can be run on these following modes:
    - Local node without ray, but only support single GPU
    - Local node with ray, support multi-nodes and multi-GPUs
    - Remote node with ray, support multi-nodes and multi-GPUs

    If run locally, will use train_func to train the model.
        Can log the training process to db if db and llm provided.
        Will reuse the db and llm from the current process.
    If run on ray, will use ray_train to train the model.
        Can log the training process to db if db and llm provided.
        Will rebuild the db and llm for the new process that can access to db.
        The ray cluster must can access to db.

    :param training_args: training Arguments, see LLMTrainingArguments
    :param train_dataset: training dataset
    :param eval_datasets: evaluation dataset, can be a dict of datasets
    :param model_kwargs: model kwargs for AutoModelForCausalLM
    :param tokenizer_kwargs: tokenizer kwargs for AutoTokenizer
    :param db: datalayer, used for creating LLMCallback
    :param llm: llm model, used for creating LLMCallback
    :param ray_configs: ray configs, must provide if using ray
    :param kwargs: other kwargs for Trainer
    """
    on_ray = bool(ray_configs)

    # Auto detect multi-GPUs and use ray to run data parallel training
    # If not todo this, will run on a bad parallel mode
    if not on_ray and torch.cuda.device_count() > 1 and training_args.num_gpus != 1:
        on_ray = True
        logging.warn("Detected multi-GPUs, will use ray to run training on multi-GPUs")

    if training_args.deepspeed:
        on_ray = True
        logging.warn("Detected deepspeed, will use ray to run training on deepspeed")

    log_to_db = training_args.log_to_db

    if not on_ray:
        # create local LLMCallback
        if db is not None and llm is not None and log_to_db:
            callbacks = [
                LLMCallback(db=db, llm=llm, experiment_id=training_args.experiment_id)
            ]
        else:
            callbacks = None
        # build training_args for local training
        training_args.build()
        return train_func(
            training_args,
            train_dataset,
            eval_datasets,
            model_kwargs,
            tokenizer_kwargs,
            callbacks=callbacks,
            **kwargs,
        )

    else:
        # create remote LLMCallback, ray cluster must can access to db
        if db is not None and llm is not None and log_to_db:
            from superduper import CFG

            cfg = getattr(db, "cfg", deepcopy(CFG))
            callbacks = [
                LLMCallback(
                    cfg=cfg,
                    identifier=llm.identifier,
                    experiment_id=training_args.experiment_id,
                )
            ]
        else:
            callbacks = None
        results = ray_train(
            training_args,
            train_dataset,
            eval_datasets,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            callbacks=callbacks,
            ray_configs=ray_configs,
            **kwargs,
        )
        logging.info(f"Training finished, results: {results}")

        if not log_to_db:
            handle_ray_results(db, llm, results)
        return results


def handle_ray_results(db, llm, results):
    """Handle the ray results.

    Will save the checkpoint to db if db and llm provided.

    :param db: datalayer, used for saving the checkpoint
    :param llm: llm model, used for saving the checkpoint
    :param results: the ray training results, contains the checkpoint
    """
    checkpoint = results.checkpoint
    if checkpoint is None:
        logging.warn("No checkpoint found, skip saving checkpoint")
        return results
    path = checkpoint.path
    if checkpoint.filesystem.type_name == "s3":
        # download the checkpoint from s3
        logging.info(f"Download checkpoint from s3, {checkpoint}")
        path = checkpoint.to_directory()
        logging.info(f"Downloaded checkpoint to {path}")

    # Pad the path to the checkpoint
    path = os.path.join(path, "checkpoint")
    if llm is not None:
        llm.adapter_id = Checkpoint(
            identifier=llm.identifier, path=path, step=int(path.split("-")[-1])
        )
        db.apply(llm.adapter_id, force=True)
        if db is not None:
            db.replace(llm, upsert=True, force=True)


def train_func(
    training_args: LLMTrainer,
    train_dataset: "Dataset",
    eval_datasets: t.Union["Dataset", t.Dict[str, "Dataset"]],
    model_kwargs: dict,
    tokenizer_kwargs: dict,
    trainer_prepare_func: t.Optional[t.Callable] = None,
    callbacks=None,
    **kwargs,
):
    """Base training function for LLM model.

    :param training_args: training Arguments, see LLMTrainingArguments
    :param train_dataset: training dataset,
        can be huggingface datasets.Dataset or ray.data.Dataset
    :param eval_datasets: evaluation dataset, can be a dict of datasets
    :param model_kwargs: model kwargs for AutoModelForCausalLM
    :param tokenizer_kwargs: tokenizer kwargs for AutoTokenizer
    :param trainer_prepare_func: function to prepare trainer
        This function will be called after the trainer is created,
        we can add some custom settings to the trainer
    :param callbacks: list of callbacks will be added to the trainer
    :param kwargs: other kwargs for Trainer
        All the kwargs will be passed to Trainer,
        make sure the Trainer support these kwargs
    """
    logging.info("Start training LLM model")
    logging.info(f"training_args: {training_args}")
    model_kwargs = deepcopy(model_kwargs)
    tokenizer_kwargs = deepcopy(tokenizer_kwargs)
    # Get device map
    device_map: t.Union[None, str, t.Dict[str, int]] = model_kwargs.get("device_map")
    if os.environ.get("LOCAL_RANK") is not None:
        ddp = int(os.environ.get("WORLD_SIZE", 1)) != 1
        device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))} if ddp else None
    elif torch.backends.mps.is_available():
        device_map = "mps"

    quantization_config = create_quantization_config(training_args)

    if is_deepspeed_zero3_enabled():
        device_map = None
        model_kwargs["low_cpu_mem_usage"] = False
        if quantization_config is not None:
            raise ValueError(
                "Quantization is not supported with ZeRO-3. Please use ZeRO-2 instead."
            )

    logging.info("Overwriting model_kwargs for LLM training")
    logging.info(f"quantization_config: {quantization_config}")
    logging.info(f"device_map: {device_map}")

    model_kwargs["quantization_config"] = quantization_config
    model_kwargs["device_map"] = device_map
    logging.info(f"model_kwargs: {model_kwargs}")
    model = AutoModelForCausalLM.from_pretrained(
        **model_kwargs,
    )
    logging.info("tokenizer_kwargs: %s", tokenizer_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    from trl import SFTConfig, setup_chat_format
    from trl.trainer import SFTTrainer

    if training_args.setup_chat_format:
        logging.info("Setup chat format")
        model, tokenizer = setup_chat_format(model, tokenizer)

    if training_args.use_lora:
        logging.info("Preparing LoRA training")
        model = prepare_lora_training(model, training_args)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args.build_training_args(build_class=SFTConfig),
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        max_seq_length=training_args.max_seq_length,
        **kwargs,
    )
    if trainer_prepare_func is not None:
        trainer = trainer_prepare_func(trainer)

    for callback in callbacks or []:
        logging.info(f"Add callback {callback}")
        trainer.add_callback(callback)
    trainer.model.config.use_cache = False
    results = trainer.train()
    trainer.save_state()
    return results


@wraps(train_func)
def ray_train(
    training_args: LLMTrainer,
    train_dataset,
    eval_datasets,
    ray_configs: t.Optional[t.Dict[str, t.Any]] = None,
    **kwargs,
):
    """Ray training function for LLM model.

    The ray train function will handle the following logic:
    - Prepare the datasets for ray
    - Build the training_loop_func for ray
    - Connect to ray cluster
    - Make some modifications to be compatible with ray finetune llm

    :param training_args: training Arguments, see LLMTrainingArguments
    :param train_dataset: training dataset,
        can be huggingface datasets.Dataset
    :param eval_datasets: evaluation dataset,
        Must be a Huggingface datasets.Dataset
    :param ray_configs: ray configs, must provide if using ray_configs
    :param **kwargs: other kwargs for Trainer
    """
    import ray
    from ray import train
    from ray.train import ScalingConfig
    from ray.train.huggingface.transformers import (
        RayTrainReportCallback,
        prepare_trainer,
    )
    from ray.train.torch import TorchTrainer

    ray.data.DataContext.get_current().execution_options.verbose_progress = True

    def trainer_prepare_func(trainer):
        # TODO: Check issues of RayTrainReportCallback run on multi-nodes
        trainer.add_callback(RayTrainReportCallback())
        trainer = prepare_trainer(trainer)
        return trainer

    def ray_train_func(train_loop_config):
        os.environ["OMP_NUM_THREADS"] = str(
            train.get_context().get_trial_resources().bundles[-1].get("CPU", 1)
        )

        logging.info(f"Start training on ray, train_dataset: {len(train_dataset)}")
        kwargs["trainer_prepare_func"] = trainer_prepare_func

        # Note: Set use_reentrant to False when using ray+lora+gradient_checkpointing
        # If not, will cause error "Varibable has been marked as ready twice"
        # Seems to be some parameter compatibility issue between ray and peft
        if train_loop_config.get(
            "gradient_checkpointing", False
        ) and train_loop_config.get("use_lora", False):
            logging.warn(
                "Using Ray + LoRA + Gradient Checkpointing, set use_reentrant to False"
            )
            gradient_checkpointing_kwargs = (
                train_loop_config.get("gradient_checkpointing_kwargs", {}) or {}
            )
            gradient_checkpointing_kwargs["use_reentrant"] = False
            train_loop_config[
                "gradient_checkpointing_kwargs"
            ] = gradient_checkpointing_kwargs
        train_loop_args = LLMTrainer(**train_loop_config)
        # Build the training_args on remote machine
        train_loop_args.build()
        return train_func(train_loop_args, train_dataset, eval_datasets, **kwargs)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    if not ray_configs:
        gpu_count = training_args.num_gpus or torch.cuda.device_count()
        ray_configs = {
            "scaling_config": ScalingConfig(
                num_workers=gpu_count or 1,
                use_gpu=bool(gpu_count),
            )
        }
        logging.warn(f"Set ray_configs to {ray_configs}")
        logging.warn("Suggest to set ray_configs manually for better performance")
    if "scaling_config" not in ray_configs:
        raise ValueError("Please provide scaling_config")

    if "run_config" not in ray_configs:
        logging.warn("No run_config provided")

    logging.info(f"Start training on ray, ray_configs: {ray_configs}")

    trainer = TorchTrainer(
        train_loop_per_worker=ray_train_func,
        train_loop_config=training_args.to_dict(),
        # Don't use ray dataset, because it is not compatible with ConstantLengthDataset
        # when running on multiple GPUs
        **ray_configs,
    )

    results = trainer.fit()
    return results


def prepare_lora_training(model, config: LLMTrainer):
    """Prepare LoRA training for the model.

    Get the LoRA target modules and convert the model to peft model.

    :param model: The model to prepare for LoRA training.
    :param config: The configuration to use.
    """
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as e:
        raise ImportError("Please install peft to use LoRA training") from e

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type="CAUSAL_LM",
    )

    if config.bits:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )

        ddp = int(os.environ.get("WORLD_SIZE", 1)) != 1
        if not ddp and torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

    model = get_peft_model(model, lora_config)

    if config.gradient_checkpointing:
        model.enable_input_require_grads()

    if config.local_rank == 0:
        model.print_trainable_parameters()
    return model


def create_quantization_config(config: LLMTrainer):
    """Create quantization config for LLM training.

    :param config: The configuration to use.
    """
    compute_dtype = (
        torch.float16
        if config.fp16
        else (torch.bfloat16 if config.bf16 else torch.float32)
    )
    if config.bits is not None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config.bits == 4,
            load_in_8bit=config.bits == 8,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quantization_config = None
    return quantization_config
