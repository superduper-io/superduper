from superduperdb.ext.llm.train import LLMTrainer

trainer = LLMTrainer(model="mistralai/mistral-7b-instruct-v0.2", bits=4, qlora=True, output_dir='~/data/llm-finetune/output/')

trainer.train()
