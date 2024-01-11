import torch
from datasets import load_dataset

from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
from superduperdb.base.document import Document
from superduperdb.ext.llm import LLM, LLMTrainingConfiguration

db = superduper("mongodb://localhost:27017/llm-finetune")

prompt_template = (
    "Below is an instruction that describes a task,"
    "paired with an input that provides further context. "
    "Write a response that appropriately completes the request."
    "\n\n### Instruction:\n{x}\n\n### Response:\n{y}"
)
#
datas = load_dataset("c-s-ale/alpaca-gpt4-data-zh")["train"].to_list()[:1000]

for data in datas:
    if data["input"] is not None:
        data["instruction"] = data["instruction"] + "\n" + data["input"]
    data["text"] = prompt_template.format(x=data["instruction"], y=data["output"])

db.execute(Collection("alpaca-gpt4-data-zh").insert_many(list(map(Document, datas))))



# training
llm = LLM(
    identifier="llm-finetune",
    bits=4,
    model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
    tokenizer_kwags={"model_max_length": 512},
)
training_configuration = LLMTrainingConfiguration(identifier="llm-finetune",
    output_dir="outputs/llm-finetune",
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    num_train_epochs=3,
    fp16=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=10,
    learning_rate=2e-5,
    weight_decay=0.0,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_strategy="steps",
    logging_steps=5,
    gradient_checkpointing=True,
    report_to=["wandb"],
)

llm.fit(
    X="text",
    db=db,
    select=Collection("alpaca-gpt4-data-zh").find(),
    configuration=training_configuration,
    prefetch_size=1000,
)



# inference
llm = LLM(
    identifier="llm-finetune",
    model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
    bits=4,
    pretrain_kwargs= dict(attn_implementation="flash_attention_2"),
    tokenizer_kwags={"model_max_length": 512},
)
db.add(llm)
llm.add_adapter("outputs/llm-finetune/checkpoint-350", "zh-350")
llm.add_adapter("outputs/llm-finetune/checkpoint-200", "zh-200")

datas = list(Collection("alpaca-gpt4-data-zh").find().execute(db))
data = datas[3].content
print(data["text"])

prompt = prompt_template.format(x=data['instruction'], y="")
print('-' * 20, '\n')
print(prompt)
print('-' * 20, '\n')
print("Base model:")
print(llm.predict(prompt, max_new_tokens=1000, one=True))

print('-' * 20, '\n')
print("Finetuned model-350:")
print(llm.predict(prompt, max_new_tokens=1000, one=True, adapter_name="zh-350"))

print('-' * 20, '\n')
print("Finetuned model-200:")
print(llm.predict(prompt, max_new_tokens=1000, one=True, adapter_name="zh-200"))
