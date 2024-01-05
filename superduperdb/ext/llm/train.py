import logging
import os
import typing
from dataclasses import dataclass, field
from typing import List, Optional, Union

import bitsandbytes as bnb
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    deepspeed,
)

if typing.TYPE_CHECKING:
    from superduperdb.backends.base.query import Select
    from superduperdb.base.datalayer import Datalayer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="mistralai/mistral-7b-instruct-v0.2"
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


@dataclass
class DataArguments:
    data_name: str = field(
        default="c-s-ale/alpaca-gpt4-data-zh",
        metadata={"help": "dataset name"},
    )


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    bits: Optional[int] = None


instruction_template = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:",
}


def create_dataset(tokenizer, data_args):
    def process_dataset(example):
        instruction = example["instruction"]
        input = example.get("input", None)
        output = example.get("output", None)

        if input is not None:
            prompt = instruction_template["prompt_input"].format(
                instruction=instruction, input=input
            )

        else:
            prompt = instruction_template["prompt_no_input"].format(
                instruction=instruction
            )

        if output is not None:
            prompt = prompt + output + tokenizer.eos_token
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = load_dataset(data_args.data_name, split="train[:3000]")

    dataset = dataset.train_test_split(
        test_size=0.1,
        seed=42,
    )
    train_dataset = dataset["train"].map(process_dataset)
    eval_dataset = dataset["test"].map(process_dataset)
    return train_dataset, eval_dataset


def find_all_linear_names(args, model):
    cls = (
        bnb.nn.Linear4bit
        if args.bits == 4
        else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


@dataclass
class LLMTrainer:
    model: str = "mistralai/mistral-7b-instruct-v0.2"
    bits: Optional[int] = None
    qlora: bool = True
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    output_dir: str = "output"
    batch_size: int = 16
    model_args: ModelArguments = field(default_factory=ModelArguments)
    data_args: DataArguments = field(default_factory=DataArguments)
    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(output_dir="")
    )
    lora_args: LoraArguments = field(default_factory=LoraArguments)

    def __post_init__(self, **kwargs):
        self.model_args.model_name_or_path = self.model
        self.lora_args.bits = self.bits
        self.lora_args.q_lora = self.qlora

        self.training_args.per_device_train_batch_size = (
            self.per_device_train_batch_size
        )
        self.training_args.per_device_eval_batch_size = self.per_device_eval_batch_size
        self.training_args.gradient_accumulation_steps = (
            self.gradient_accumulation_steps
        )
        self.training_args.output_dir = self.output_dir

    def _fit(self, X, Y, db: "Datalayer", select: "Select"):
        from superduperdb.backends.query_dataset import QueryDataset

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            use_fast=False,
            model_max_length=self.training_args.model_max_length,
            padding_side="left",
            trust_remote_code=self.model_args.trust_remote_code,
        )
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        def process_dataset(example):
            instruction = example[X]
            input = example.get("input", None)
            output = example.get(Y, None)

            if input is not None:
                prompt = instruction_template["prompt_input"].format(
                    instruction=instruction, input=input
                )

            else:
                prompt = instruction_template["prompt_no_input"].format(
                    instruction=instruction
                )

            if output is not None:
                prompt = prompt + output + tokenizer.eos_token
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result

        self.train_dataset = QueryDataset(select=select.find({"_fold": "valid"}), db=db)
        self.dev_dataset = QueryDataset(select=select.find({"_fold": "valid"}), db=db)
        self.train_dataset = self.train_dataset.map(process_dataset)
        self.eval_dataset = self.dev_dataset.map(process_dataset)
        self.train()

    def train(self):
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1

        compute_dtype = (
            torch.float16
            if self.training_args.fp16
            else (torch.bfloat16 if self.training_args.bf16 else torch.float32)
        )

        if self.lora_args.bits is not None:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.lora_args.bits == 4,
                load_in_8bit=self.lora_args.bits == 8,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
            if (
                len(self.training_args.fsdp) > 0
                or deepspeed.is_deepspeed_zero3_enabled()
            ):
                logging.warning(
                    "FSDP and ZeRO3 are both currently incompatible with QLoRA."
                )
        else:
            quantization_config = None
            device_map = None

        model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            quantization_config=quantization_config,
            low_cpu_mem_usage=not deepspeed.is_deepspeed_zero3_enabled(),
            device_map=device_map,
            trust_remote_code=self.model_args.trust_remote_code,
        )

        target_modules = self.lora_args.lora_target_modules or find_all_linear_names(
            self.lora_args, model
        )

        lora_config = LoraConfig(
            r=self.lora_args.lora_r,
            lora_alpha=self.lora_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_args.lora_dropout,
            bias=self.lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if self.lora_args.bits:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.training_args.gradient_checkpointing,
            )

            if not ddp and torch.cuda.device_count() > 1:
                model.is_parallelizable = True
                model.model_parallel = True

        model = get_peft_model(model, lora_config)

        if (
            self.training_args.deepspeed is not None
            and self.training_args.local_rank == 0
        ):
            model.print_trainable_parameters()

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        if self.training_args.local_rank == 0:
            model.print_trainable_parameters()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            use_fast=False,
            model_max_length=self.training_args.model_max_length,
            padding_side="left",
            trust_remote_code=self.model_args.trust_remote_code,
        )

        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        trainer = transformers.Trainer(
            model=model,
            tokenizer=tokenizer,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )
        model.config.use_cache = False

        trainer.train()
        trainer.save_state()
