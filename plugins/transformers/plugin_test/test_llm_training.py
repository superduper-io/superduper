import dataclasses as dc
import os
import random

from superduper.base.base import Base
from superduper.components.dataset import Dataset
from superduper.components.metric import Metric

from superduper_transformers import LLM

TEST_MODEL_NAME = "facebook/opt-125m"
try:
    import datasets
    import peft
    import trl

    from superduper_transformers.training import LLMTrainer
except ImportError:
    datasets = None
    peft = None
    trl = None


def create_fold():
    import random

    if random.random() < 0.2:
        return 'valid'
    return 'train'


class documents(Base):
    text: str
    id: str
    _fold: str = dc.field(default_factory=create_fold)


def test_training(db, tmpdir):

    db.create(documents)

    datas = []
    for i in range(32 + 8):
        text = f"{i}+1={i+1}"
        fold = "train" if i < 32 else "valid"
        datas.append({"text": text, "id": str(i), "_fold": fold})

    collection = db['documents']
    collection.insert(datas)
    select = collection.select()

    transform = None
    key = "text"
    training_kwargs = dict(dataset_text_field="text")

    trainer = LLMTrainer(
        identifier="llm-finetune",
        output_dir=str(tmpdir),
        num_train_epochs=1,
        save_total_limit=3,
        logging_steps=1,
        eval_steps=1,
        save_strategy="steps",
        evaluation_strategy="steps",
        save_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        max_seq_length=64,
        use_lora=True,
        gradient_checkpointing=True,
        key=key,
        select=select,
        transform=transform,
        training_kwargs=training_kwargs,
        report_to=[],
    )

    def metric(predictions, targets):
        return random.random()

    validation_sets = [
        Dataset(
            identifier="dataset_1",
            select=collection.filter(collection['_fold'] == "valid"),
        ),
        Dataset(
            identifier="dataset_2",
            select=collection.filter(collection['_fold'] == "valid"),
        ),
    ]
    metrics = [
        Metric(
            identifier="metrics1",
            object=metric,
        ),
        Metric(
            identifier="metrics2",
            object=metric,
        ),
    ]

    from superduper import Validation

    validation = Validation(
        identifier="validation",
        datasets=validation_sets,
        metrics=metrics,
        key='text',
    )

    model = LLM(
        identifier="llm",
        model_name_or_path="facebook/opt-125m",
        tokenizer_kwargs=dict(model_max_length=64),
        trainer=trainer,
        validation=validation,
    )

    db.apply(model)

    # Load from db directly
    llm = db.load("LLM", "llm")
    assert isinstance(llm.predict("1+1="), str)

    # load from checkpoint
    experiment_id = db.show("Checkpoint")[-1]
    checkpoint = db.load("Checkpoint", experiment_id)

    llm = LLM(
        identifier="llm_checkpoint",
        adapter_id=checkpoint,
        model_name_or_path="facebook/opt-125m",
        tokenizer_kwargs=dict(model_max_length=64),
    )
    llm.db = db
    assert isinstance(llm.predict("1+1="), str)

    # load from local path
    checkpoint_paths = os.listdir(tmpdir)
    checkpoint_paths = [
        checkpoint
        for checkpoint in checkpoint_paths
        if checkpoint.startswith("checkpoint")
    ]
    checkpoint_paths = sorted(checkpoint_paths, key=lambda x: int(x.split("-")[-1]))
    # Test checkpoints
    assert len(checkpoint_paths) == 3
    for checkpoint_path in checkpoint_paths:
        llm = LLM(
            identifier=checkpoint_path,
            adapter_id=os.path.join(tmpdir, checkpoint_path),
            model_name_or_path="facebook/opt-125m",
            tokenizer_kwargs=dict(model_max_length=64),
        )
        assert isinstance(llm.predict("1+1="), str)
