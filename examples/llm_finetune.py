import os

import torch
from datasets import load_dataset

from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
from superduperdb.base.document import Document
from superduperdb.ext.llm.model import LLM, LLMTrainingConfiguration

prompt_template = (
    "Below is an instruction that describes a task,"
    "paired with an input that provides further context. "
    "Write a response that appropriately completes the request."
    "\n\n### Instruction:\n{x}\n\n### Response:\n{y}"
)

collection_name = "alpaca-gpt4-data-zh"


def prepare_datas(db, size):
    datas = load_dataset("c-s-ale/alpaca-gpt4-data-zh")["train"].to_list()[:size]

    for data in datas:
        if data["input"] is not None:
            data["instruction"] = data["instruction"] + "\n" + data["input"]
        data["text"] = prompt_template.format(x=data["instruction"], y=data["output"])

    db.execute(Collection(collection_name).insert_many(list(map(Document, datas))))


def train(db, model_identifier, model_name, output_dir):
    # training
    llm = LLM(
        identifier=model_identifier,
        bits=4 if torch.cuda.is_available() else None,
        model_name_or_path=model_name,
    )
    training_configuration = LLMTrainingConfiguration(
        identifier="llm-finetune-training-config",
        output_dir=output_dir,
        overwrite_output_dir=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        # num_train_epochs=1,
        max_steps=10,
        fp16=torch.cuda.is_available(),  # mps don't support fp16
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=3,
        learning_rate=2e-5,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=5,
        gradient_checkpointing=True,
        report_to=[],
        on_ray=False,
    )

    llm.fit(
        X="text",
        db=db,
        select=Collection(collection_name).find(),
        configuration=training_configuration,
        prefetch_size=1000,
    )


def inference(db, model_identifier, output_dir):
    # inference
    llm_base = db.load("model", model_identifier)
    checkpoints = [
        checkpoint
        for checkpoint in os.listdir(output_dir)
        if checkpoint.startswith("checkpoint")
    ]
    db.add(llm_base)
    for checkpoint in checkpoints:
        llm_checkpoint = LLM(
            identifier=checkpoint,
            bits=4 if torch.cuda.is_available() else None,
            adapter_id=os.path.join(output_dir, checkpoint),
            model_name_or_path=llm_base.model_name_or_path,
        )
        db.add(llm_checkpoint)

    datas = list(Collection(collection_name).find().execute(db))
    data = datas[3].content
    print(data["text"])

    prompt = prompt_template.format(x=data["instruction"], y="")
    print("-" * 20, "\n")
    print(prompt)
    print("-" * 20, "\n")

    print("Base model:\n")
    print(db.predict(llm_base.identifier, prompt, max_new_tokens=100))

    for checkpoint in checkpoints:
        print("-" * 20, "\n")
        print(f"Finetuned model-{checkpoint}:\n")
        print(db.predict(checkpoint, prompt, max_new_tokens=100))


if __name__ == "__main__":
    db = superduper("mongomock://localhost:27017/test-llm")
    model = "facebook/opt-125m"
    output_dir = "outputs/llm-finetune"

    db.drop(force=True)
    prepare_datas(db, size=200)
    train(db, "llm-finetune", model, output_dir)
    inference(db, "llm-finetune", output_dir)
