---
sidebar_label: Build A Trainable LLM
filename: build_a_trainable_llm.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Build A Trainable LLM

**Create an LLM Trainer for training**

The parameters of this LLM Trainer are basically the same as `transformers.TrainingArguments`, but some additional parameters have been added for easier training setup.

```python
from superduperdb.ext.transformers import LLM, LLMTrainer
trainer = LLMTrainer(
    identifier="llm-finetune-trainer",
    output_dir="output/finetune",
    overwrite_output_dir=True,
    num_train_epochs=3,
    save_total_limit=3,
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=100,
    eval_steps=100,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    max_seq_length=512,
    key=key,
    select=select,
    transform=transform,
    training_kwargs=training_kwargs,
)
```


<Tabs>
    <TabItem value="Lora" label="Lora" default>
        ```python
        trainer.use_lora = True        
        ```
    </TabItem>
    <TabItem value="QLora" label="QLora" default>
        ```python
        trainer.use_lora = True
        trainer.bits = 4        
        ```
    </TabItem>
    <TabItem value="Deepspeed" label="Deepspeed" default>
        ```python
        !pip install deepspeed
        deepspeed = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "zero_optimization": {
                "stage": 2,
            },
        }
        trainer.use_lora = True
        trainer.bits = 4
        trainer.deepspeed = deepspeed        
        ```
    </TabItem>
    <TabItem value="Multi-GPUS" label="Multi-GPUS" default>
        ```python
        trainer.use_lora = True
        trainer.bits = 4
        trainer.num_gpus = 2        
        ```
    </TabItem>
</Tabs>
Create a trainable LLM model and add it to the database, then the training task will run automatically.

```python
llm = LLM(
    identifier="llm",
    model_name_or_path=model_name,
    trainer=trainer,
    model_kwargs=model_kwargs,
    tokenizer_kwargs=tokenizer_kwargs,
)

db.apply(llm)
```

# Load the trained model
There are two methods to load a trained model:

- **Load the model directly**: This will load the model with the best metrics (if the transformers' best model save strategy is set) or the last version of the model.
- **Use a specified checkpoint**: This method downloads the specified checkpoint, then initializes the base model, and finally merges the checkpoint with the base model. This approach supports custom operations such as resetting flash_attentions, model quantization, etc., during initialization.


<Tabs>
    <TabItem value="Load Trained Model Directly" label="Load Trained Model Directly" default>
        ```python
        llm = db.load("model", "llm")        
        ```
    </TabItem>
    <TabItem value="Use a specified checkpoint" label="Use a specified checkpoint" default>
        ```python
        from superduperdb.ext.transformers import LLM, LLMTrainer
        experiment_id = db.show("checkpoint")[-1]
        version = None # None means the last checkpoint
        checkpoint = db.load("checkpoint", experiment_id, version=version)
        llm = LLM(
            identifier="llm",
            model_name_or_path=model_name,
            adapter_id=checkpoint,
            model_kwargs=dict(load_in_4bit=True)
        )        
        ```
    </TabItem>
</Tabs>
<DownloadButton filename="build_a_trainable_llm.md" />
