from datasets import load_dataset

from superduperdb import superduper
from superduperdb.ext.llm.model import LLM
from superduperdb.ext.llm.training import LLMTrainer

db = superduper("mongodb://localhost:27017/llm-finetune")

dataset = load_dataset("philschmid/dolly-15k-oai-style")

dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

llm = LLM(
    identifier="llm-finetune",
    model_name_or_path="mistralai/Mistral-7B-v0.1",
)

trainer = LLMTrainer(
    identifier="llm-finetune-training-config",
    output_dir="output/dolly-chatml",
    learning_rate=0.0002,
    lr_scheduler_type='constant',
    warmup_ratio=0.003,
    max_grad_norm=0.3,
    overwrite_output_dir=True,
    num_train_epochs=3,
    save_total_limit=3,
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=200,
    eval_steps=200,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    max_seq_length=512,
    use_lora=True,
    log_to_db=True,
    fp16=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    bits=4,
)

llm.trainer = trainer

llm.fit(
    train_dataset=train_dataset,
    valid_dataset=eval_dataset,
    db=db,
)


llm = db.load("model", "llm-finetune")

messages = [
    {
        "role": "user",
        "content": "What is the capital of Germany? Explain why thats the case and if it was different in the past?",
    }
]
print(llm.predict_one(messages, max_new_tokens=200, do_sample=False))
