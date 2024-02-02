from datasets import load_dataset

from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
from superduperdb.base.document import Document
from superduperdb.ext.llm import LLM
from superduperdb.ext.llm.model import LLMTrainingConfiguration

model = "mistralai/Mistral-7B-v0.1"
dataset_name = "timdettmers/openassistant-guanaco"

db = superduper("mongomock://test_llm")
dataset = load_dataset(dataset_name)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

train_documents = [
    Document({"text": example["text"], "_fold": "train"}) for example in train_dataset
]
eval_documents = [
    Document({"text": example["text"], "_fold": "valid"}) for example in eval_dataset
]

db.execute(Collection("datas").insert_many(train_documents))
db.execute(Collection("datas").insert_many(eval_documents))

llm = LLM(
    identifier="llm-finetune",
    bits=4,
    model_name_or_path=model,
)


training_configuration = LLMTrainingConfiguration(
    identifier="llm-finetune-training-config",
    output_dir="output/llm-finetune",
    overwrite_output_dir=True,
    num_train_epochs=1,
    save_total_limit=5,
    logging_steps=10,
    evaluation_strategy="steps",
    fp16=True,
    eval_steps=100,
    save_steps=100,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    max_length=512,
    use_lora=True,
)

llm.fit(
    X="text",
    select=Collection("datas").find(),
    configuration=training_configuration,
    db=db,
)


prompt = "### Human: Who are you? ### Assistant: "

# Automatically load lora model for prediction, default use the latest checkpoint
print(llm.predict(prompt, max_new_tokens=100, do_sample=True))
