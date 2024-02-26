from datasets import load_dataset

from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
from superduperdb.base.document import Document
from superduperdb.ext.llm import LLM
from superduperdb.ext.llm.model import LLMTrainingConfiguration

model = "facebook/opt-125m"
dataset_name = "imdb"
train_dataset = load_dataset("imdb", split="train")

db = superduper("mongomock://test_llm")

llm = LLM(
    identifier="llm-finetune",
    model_name_or_path=model,
)


training_configuration = LLMTrainingConfiguration(
    identifier="llm-finetune-training-config",
    output_dir="output/llm-finetune",
    overwrite_output_dir=True,
    num_train_epochs=1,
    save_total_limit=5,
    logging_steps=1,
    # evaluation_strategy="steps",
    # eval_steps=10,
    save_steps=100,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    max_seq_length=512,
    use_lora=True,
    log_to_db=True,
)

llm.fit(
    train_dataset=train_dataset,
    X="text",
    configuration=training_configuration,
    db=db,
)


prompt = "### Human: Who are you? ### Assistant: "

# Automatically load lora model for prediction, default use the latest checkpoint
print(llm.predict_one(prompt))
