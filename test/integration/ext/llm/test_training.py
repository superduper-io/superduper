import os

import pytest
import transformers

from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
from superduperdb.base.document import Document
from superduperdb.ext.llm.model import LLM, LLMTrainingConfiguration

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# Don't run on CI
RUN_LLM_FINETUNE = os.environ.get("RUN_LLM_FINETUNE", "0") == "1"

# Some predefined parameters
# model = "facebook/opt-350m"
model = "mistralai/Mistral-7B-v0.1"
dataset_name = "timdettmers/openassistant-guanaco"
prompt = "### Human: Who are you? ### Assistant: "

save_folder = "output"


@pytest.fixture
def db():
    db_ = superduper("mongomock://localhost:30000/test_llm")
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

    db_.execute(Collection("datas").insert_many(train_documents))
    db_.execute(Collection("datas").insert_many(eval_documents))

    yield db_

    db_.drop(force=True)


@pytest.fixture
def base_config():
    return LLMTrainingConfiguration(
        identifier="llm-finetune-training-config",
        overwrite_output_dir=True,
        num_train_epochs=1,
        save_total_limit=5,
        logging_steps=10,
        evaluation_strategy="steps",
        fp16=True,
        eval_steps=200,
        save_steps=200,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        log_to_db=True,
        max_seq_length=512,
        use_lora=True,
    )


@pytest.mark.skipif(not RUN_LLM_FINETUNE, reason="RUN_LLM_FINETUNE is not set")
def test_full_finetune(db, base_config):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
    )

    base_config.kwargs["use_lora"] = False
    # Don't log to db if full finetune cause the large files
    base_config.kwargs["log_to_db"] = False
    output_dir = os.path.join(save_folder, "test_full_finetune")
    base_config.kwargs["output_dir"] = output_dir

    llm.fit(
        X="text",
        select=Collection("datas").find(),
        configuration=base_config,
        db=db,
    )

    llm_inference = LLM(
        identifier="llm",
        model_name_or_path=transformers.trainer.get_last_checkpoint(output_dir),
        model_kwargs=dict(device_map="auto"),
    )
    db.add(llm_inference)
    result = db.predict("llm", prompt, max_new_tokens=100, do_sample=False)[0].content
    print(result)


@pytest.mark.skipif(not RUN_LLM_FINETUNE, reason="RUN_LLM_FINETUNE is not set")
def test_lora_finetune(db, base_config):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
    )

    output_dir = os.path.join(save_folder, "test_lora_finetune")
    base_config.kwargs["output_dir"] = output_dir

    llm.fit(
        X="text",
        select=Collection("datas").find(),
        configuration=base_config,
        db=db,
    )

    assert os.path.exists(llm.adapter_id)

    result = db.predict("llm-finetune", prompt, max_new_tokens=100, do_sample=False)[
        0
    ].content
    assert len(result) > 0


@pytest.mark.skipif(not RUN_LLM_FINETUNE, reason="RUN_LLM_FINETUNE is not set")
def test_qlora_finetune(db, base_config):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
    )

    base_config.kwargs["bits"] = 4
    output_dir = os.path.join(save_folder, "test_qlora_finetune")
    base_config.kwargs["output_dir"] = output_dir

    llm.fit(
        X="text",
        select=Collection("datas").find(),
        configuration=base_config,
        db=db,
    )

    assert os.path.exists(llm.adapter_id)

    result = db.predict("llm-finetune", prompt, max_new_tokens=100, do_sample=False)[
        0
    ].content
    print(result)
    assert len(result) > 0


@pytest.mark.skipif(not RUN_LLM_FINETUNE, reason="RUN_LLM_FINETUNE is not set")
def test_local_ray_lora_finetune(db, base_config):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
    )

    base_config.kwargs["use_lora"] = True
    base_config.kwargs["log_to_db"] = False
    output_dir = os.path.join(save_folder, "test_local_ray_lora_finetune")
    base_config.kwargs["output_dir"] = output_dir

    from ray.train import RunConfig, ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
    )

    run_config = RunConfig(
        storage_path=os.path.abspath(output_dir),
    )

    ray_configs = {
        "scaling_config": scaling_config,
        "run_config": run_config,
    }

    llm.fit(
        X="text",
        select=Collection("datas").find(),
        configuration=base_config,
        db=db,
        on_ray=True,
        ray_configs=ray_configs,
    )

    assert os.path.exists(llm.adapter_id)

    result = db.predict("llm-finetune", prompt, max_new_tokens=100, do_sample=False)[
        0
    ].content
    print(result)
    assert len(result) > 0


@pytest.mark.skipif(not RUN_LLM_FINETUNE, reason="RUN_LLM_FINETUNE is not set")
def test_local_ray_deepspeed_lora_finetune(db, base_config):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
    )

    deepspeed = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "zero_optimization": {
            "stage": 0,
        },
    }

    base_config.kwargs["use_lora"] = True
    base_config.kwargs["log_to_db"] = False
    output_dir = os.path.join(save_folder, "test_local_ray_deepspeed_lora_finetune")
    base_config.kwargs["output_dir"] = output_dir
    base_config.kwargs["deepspeed"] = deepspeed
    base_config.kwargs["bits"] = 4

    from ray.train import RunConfig, ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=4,
        use_gpu=True,
    )

    run_config = RunConfig(
        storage_path=os.path.abspath(output_dir),
    )

    ray_configs = {
        "scaling_config": scaling_config,
        "run_config": run_config,
    }

    llm.fit(
        X="text",
        select=Collection("datas").find(),
        configuration=base_config,
        db=db,
        on_ray=True,
        ray_configs=ray_configs,
    )

    assert os.path.exists(llm.adapter_id)

    result = db.predict("llm-finetune", prompt, max_new_tokens=100, do_sample=False)[
        0
    ].content
    print(result)
    assert len(result) > 0


@pytest.mark.skipif(not RUN_LLM_FINETUNE, reason="RUN_LLM_FINETUNE is not set")
def test_remote_ray_lora_finetune(db, base_config):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
    )

    base_config.kwargs["use_lora"] = True
    base_config.kwargs["log_to_db"] = False
    # Use absolute path, because the ray will run in remote
    output_dir = "test_ray_lora_finetune"
    base_config.kwargs["output_dir"] = output_dir

    from ray.train import RunConfig, ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
    )

    run_config = RunConfig(
        storage_path="s3://llm-test-jalon/llm-finetune",
        name="llm-finetune-test100",
    )

    ray_configs = {
        "scaling_config": scaling_config,
        "run_config": run_config,
    }

    llm.fit(
        X="text",
        select=Collection("datas").find(),
        configuration=base_config,
        db=db,
        on_ray=True,
        ray_address="ray://localhost:10001",
        ray_configs=ray_configs,
    )

    assert os.path.exists(llm.adapter_id)

    result = db.predict("llm-finetune", prompt, max_new_tokens=100, do_sample=False)[
        0
    ].content
    print(result)
    assert len(result) > 0


@pytest.mark.skipif(not RUN_LLM_FINETUNE, reason="RUN_LLM_FINETUNE is not set")
def test_remote_ray_qlora_deepspeed_finetune(db, base_config):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
    )

    deepspeed = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "zero_optimization": {
            "stage": 2,
        },
    }

    base_config.kwargs["use_lora"] = True
    base_config.kwargs["log_to_db"] = False
    # Use absolute path, because the ray will run in remote
    output_dir = "test_remote_ray_qlora_deepspeed_finetune"
    base_config.kwargs["output_dir"] = output_dir
    base_config.kwargs["deepspeed"] = deepspeed

    from ray.train import RunConfig, ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
    )

    run_config = RunConfig(
        storage_path="s3://llm-test/llm-finetune",
        name="llm-finetune-test100",
    )

    ray_configs = {
        "scaling_config": scaling_config,
        "run_config": run_config,
    }

    llm.fit(
        X="text",
        select=Collection("datas").find(),
        configuration=base_config,
        db=db,
        on_ray=True,
        ray_address="ray://localhost:10001",
        ray_configs=ray_configs,
    )

    assert os.path.exists(llm.adapter_id)

    result = db.predict("llm-finetune", prompt, max_new_tokens=100, do_sample=False)[
        0
    ].content
    print(result)
    assert len(result) > 0
