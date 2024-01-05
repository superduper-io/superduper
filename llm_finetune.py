from datasets import load_dataset

from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
from superduperdb.base.document import Document
from superduperdb.ext.llm.train import LLMTrainer

datas = load_dataset("c-s-ale/alpaca-gpt4-data-zh")["train"].to_list()[:1000]


db = superduper("mongomock://llm-finetune")

db.execute(Collection("doc").insert_many(list(map(Document, datas))))


trainer = LLMTrainer(
    model="mistralai/mistral-7b-instruct-v0.2",
    bits=4,
    qlora=True,
    output_dir="~/data/llm-finetune/output/",
)

trainer._fit('instruction', 'output', db=db, select=Collection('doc'))
