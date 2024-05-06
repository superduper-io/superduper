import os

import pytest
import transformers

from superduperdb import superduper
from superduperdb.backends.mongodb import MongoQuery
from superduperdb.base.document import Document
from superduperdb.ext.transformers.model import LLM
from superduperdb.ext.transformers.training import LLMTrainer

try:
    import datasets
    import peft
    import trl
except ImportError:
    datasets = None
    peft = None
    trl = None


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

    db_.execute(MongoQuery("datas").insert_many(train_documents[:100]))
    db_.execute(MongoQuery("datas").insert_many(eval_documents[:4]))

    yield db_

    db_.drop(force=True)


@pytest.fixture
def trainer():
    return LLMTrainer(
        identifier="llm-finetune-training-config",
        overwrite_output_dir=True,
        num_train_epochs=1,
        max_steps=5,
        save_total_limit=5,
        logging_steps=10,
        evaluation_strategy="steps",
        fp16=True,
        eval_steps=1,
        save_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        log_to_db=True,
        max_seq_length=512,
        use_lora=True,
        key="text",
        select=Collection("datas").find(),
    )


@pytest.mark.skipif(
    not RUN_LLM_FINETUNE, reason="The peft, datasets and trl are not installed"
)
def test_full_finetune(db, trainer):
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

<<<<<<< HEAD
    db.apply(llm)
=======
    llm.train_X = 'text'
    llm.train_select = MongoQuery('datas').find()
    llm.trainer = trainer

    llm.fit_in_db(db=db)
>>>>>>> 9d83d21ec (Deprecate Serializable)

    llm = LLM(
        identifier="llm",
        model_name_or_path=transformers.trainer.get_last_checkpoint(output_dir),
        model_kwargs=dict(device_map="auto"),
    )

    result = llm.predict_one(prompt, max_new_tokens=200, do_sample=False)
    assert len(result) > 0


@pytest.mark.skipif(
    not RUN_LLM_FINETUNE, reason="The peft, datasets and trl are not installed"
)
def test_lora_finetune(db, trainer):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
<<<<<<< HEAD
=======
        train_X='text',
        train_select=MongoQuery('datas').find(),
>>>>>>> 9d83d21ec (Deprecate Serializable)
        trainer=trainer,
    )

    output_dir = os.path.join(save_folder, "test_lora_finetune")
    trainer.output_dir = output_dir

    db.apply(llm)

    llm = db.load("model", "llm-finetune")

    result = llm.predict_one(prompt, max_new_tokens=200, do_sample=False)
    assert len(result) > 0


@pytest.mark.skipif(
    not RUN_LLM_FINETUNE, reason="The peft, datasets and trl are not installed"
)
def test_qlora_finetune(db, trainer):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
        trainer=trainer,
<<<<<<< HEAD
=======
        train_select=MongoQuery('datas').find(),
>>>>>>> 9d83d21ec (Deprecate Serializable)
    )

    trainer.bits = 4
    output_dir = os.path.join(save_folder, "test_qlora_finetune")
    trainer.output_dir = output_dir

<<<<<<< HEAD
    db.apply(llm)
=======
    llm.fit_in_db(db=db)

    llm = db.load("model", "llm-finetune")

    result = llm.predict_one(prompt, max_new_tokens=200, do_sample=False)
    assert len(result) > 0


@pytest.mark.skipif(
    not RUN_LLM_FINETUNE, reason="The peft, datasets and trl are not installed"
)
def test_local_ray_lora_finetune(db, trainer):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
        train_X='text',
        train_select=MongoQuery('datas').find(),
        trainer=trainer,
    )

    trainer.use_lora = True
    trainer.log_to_db = False
    output_dir = os.path.join(save_folder, "test_local_ray_lora_finetune")
    trainer.output_dir = False

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
    trainer.on_ray = True
    trainer.ray_configs = ray_configs

    llm.fit_in_db(db=db)
>>>>>>> 9d83d21ec (Deprecate Serializable)

    llm = db.load("model", "llm-finetune")

    result = llm.predict_one(prompt, max_new_tokens=200, do_sample=False)
    assert len(result) > 0


@pytest.mark.skipif(
    not RUN_LLM_FINETUNE, reason="The peft, datasets and trl are not installed"
)
def test_local_ray_deepspeed_lora_finetune(db, trainer):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
<<<<<<< HEAD
=======
        train_X='text',
        train_select=MongoQuery('datas').find(),
>>>>>>> 9d83d21ec (Deprecate Serializable)
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

<<<<<<< HEAD
    db.apply(llm)
=======
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
    trainer.on_ray = True
    trainer.ray_configs = ray_configs

    llm.fit_in_db(db=db)

    llm = db.load("model", "llm-finetune")

    result = llm.predict_one(prompt, max_new_tokens=200, do_sample=False)
    assert len(result) > 0


@pytest.mark.skipif(
    not RUN_LLM_FINETUNE, reason="The peft, datasets and trl are not installed"
)
def test_remote_ray_lora_finetune(db, trainer):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
        train_X='text',
        train_select=MongoQuery('datas').find(),
        trainer=trainer,
    )

    trainer.use_lora = True
    trainer.log_to_db = False
    # Use absolute path, because the ray will run in remote
    output_dir = "test_ray_lora_finetune"
    trainer.output_dir = output_dir

    from ray.train import RunConfig, ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
    )

    run_config = RunConfig(
        storage_path="s3://llm-test/llm-finetune",
        name="llm-finetune-test",
    )

    ray_configs = {
        "scaling_config": scaling_config,
        "run_config": run_config,
    }

    trainer.on_ray = True
    trainer.ray_configs = ray_configs
    trainer.ray_address = "ray://localhost:10001"

    llm.fit_in_db(db=db)

    llm = db.load("model", "llm-finetune")

    result = llm.predict_one(prompt, max_new_tokens=200, do_sample=False)
    assert len(result) > 0


@pytest.mark.skipif(
    not RUN_LLM_FINETUNE, reason="The peft, datasets and trl are not installed"
)
def test_remote_ray_qlora_deepspeed_finetune(db, trainer):
    llm = LLM(
        identifier="llm-finetune",
        model_name_or_path=model,
        train_X='text',
        train_select=MongoQuery('datas').find(),
        trainer=trainer,
    )

    deepspeed = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "zero_optimization": {
            "stage": 2,
        },
    }

    trainer.use_lora = True
    trainer.log_to_db = False
    # Use absolute path, because the ray will run in remote
    output_dir = "test_remote_ray_qlora_deepspeed_finetune"
    trainer.output_dir = output_dir
    trainer.deepspeed = deepspeed

    from ray.train import RunConfig, ScalingConfig

    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
    )

    run_config = RunConfig(
        storage_path="s3://llm-test/llm-finetune",
        name="llm-finetune",
    )

    ray_configs = {
        "scaling_config": scaling_config,
        "run_config": run_config,
    }

    trainer.on_ray = True
    trainer.ray_configs = ray_configs
    trainer.ray_address = "ray://localhost:10001"

    llm.fit_in_db(db=db)
>>>>>>> 9d83d21ec (Deprecate Serializable)

    llm = db.load("model", "llm-finetune")

    result = llm.predict_one(prompt, max_new_tokens=200, do_sample=False)
    assert len(result) > 0
