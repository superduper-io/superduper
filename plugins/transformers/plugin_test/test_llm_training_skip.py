import os

import pytest
import transformers
from superduper import superduper
from superduper.base.document import Document

from superduper_transformers.model import LLM

try:
    import datasets
    import peft
    import torch
    import trl

    from superduper_transformers.training import LLMTrainer

    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    datasets = None
    peft = None
    trl = None
    GPU_AVAILABLE = False

# commented out due to a versioning conflict
RUN_LLM_FINETUNE = all([datasets, peft, trl])


# Some predefined parameters
model = "facebook/opt-125m"
# model = "mistralai/Mistral-7B-v0.1"
dataset_name = "timdettmers/openassistant-guanaco"
prompt = "### Human: Who are you? ### Assistant: "

save_folder = "output"


@pytest.fixture
def db():
    db_ = superduper("mongomock://localhost:30000/test_llm")
    from datasets import load_dataset

    dataset = load_dataset(dataset_name)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    train_documents = [
        Document({"text": example["text"], "_fold": "train"})
        for example in train_dataset
    ]
    eval_documents = [
        Document({"text": example["text"], "_fold": "valid"})
        for example in eval_dataset
    ]

    db_['datas'].insert_many(train_documents[:100]).execute()
    db_['datas'].insert_many(eval_documents[:4]).execute()

    yield db_

    db_.drop(force=True)


def get_trainer(db):
    return LLMTrainer(
        identifier="llm-finetune-training-config",
        overwrite_output_dir=True,
        num_train_epochs=1,
        max_steps=5,
        save_total_limit=3,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=1,
        save_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        log_to_db=True,
        max_seq_length=512,
        use_lora=True,
        key="text",
        select=db["datas"],
        training_kwargs=dict(dataset_text_field="text"),
    )


@pytest.mark.skip(reason="Maintaince going on in huggingface datasets")
@pytest.mark.skipif(
    not RUN_LLM_FINETUNE, reason="The peft, datasets and trl are not installed"
)
def test_full_finetune(db):

    trainer = get_trainer(db)
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
        trainer=trainer,
    )

    trainer.use_lora = False
    # Don't log to db if full finetune cause the large files
    trainer.log_to_db = False
    output_dir = os.path.join(save_folder, "test_full_finetune")
    trainer.output_dir = output_dir

    db.apply(llm)

    llm = LLM(
        identifier="llm",
        model_name_or_path=transformers.trainer.get_last_checkpoint(output_dir),
        model_kwargs=dict(device_map="auto"),
    )

    result = llm.predict(prompt, max_new_tokens=200, do_sample=False)
    assert len(result) > 0


@pytest.mark.skip(reason="Maintaince going on in huggingface datasets")
@pytest.mark.skipif(
    not RUN_LLM_FINETUNE, reason="The peft, datasets and trl are not installed"
)
def test_lora_finetune(db):
    trainer = get_trainer(db)
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
        trainer=trainer,
    )

    output_dir = os.path.join(save_folder, "test_lora_finetune")
    trainer.output_dir = output_dir

    db.apply(llm)

    llm = db.load("model", "llm-finetune")

    result = llm.predict(prompt, max_new_tokens=200, do_sample=False)
    assert len(result) > 0


@pytest.mark.skip(reason="Maintaince going on in huggingface datasets")
@pytest.mark.skipif(
    not (RUN_LLM_FINETUNE and GPU_AVAILABLE),
    reason="The peft, datasets and trl are not installed",
)
def test_qlora_finetune(db):

    trainer = get_trainer(db)

    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
        trainer=trainer,
    )

    trainer.bits = 4
    output_dir = os.path.join(save_folder, "test_qlora_finetune")
    trainer.output_dir = output_dir

    db.apply(llm)

    llm = db.load("model", "llm-finetune")

    result = llm.predict(prompt, max_new_tokens=200, do_sample=False)
    assert len(result) > 0


@pytest.mark.skip(reason="Maintaince going on in huggingface datasets")
@pytest.mark.skipif(
    not (RUN_LLM_FINETUNE and GPU_AVAILABLE), reason="Deepspeed need GPU"
)
def test_local_ray_deepspeed_lora_finetune(db):

    trainer = get_trainer(db)

    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
        trainer=trainer,
    )

    deepspeed = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "zero_optimization": {
            "stage": 0,
        },
    }

    trainer.use_lora = True
    output_dir = os.path.join(save_folder, "test_local_ray_deepspeed_lora_finetune")
    trainer.output_dir = output_dir
    trainer.deepspeed = deepspeed
    trainer.bits = 4

    db.apply(llm)

    llm = db.load("model", "llm-finetune")

    result = llm.predict(prompt, max_new_tokens=200, do_sample=False)
    assert len(result) > 0
