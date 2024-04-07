---
sidebar_label: Finetune LLM
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Finetune LLM

## Finetuning


<Tabs>
    <TabItem value="Local" label="Local" default>
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
            save_steps=50,
            eval_steps=50,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            max_seq_length=512,
            use_lora=True, 
            bits=4,
        )
        
        llm = LLM(
            identifier="llm",
            model_name_or_path=model_name,
            trainer=trainer,
            train_X="text", # Which field of data is used for training
            train_select=Collection('datas').find() # Which table data to use for training
        )
        
        llm.fit_in_db(db=db)        
        ```
    </TabItem>
    <TabItem value="On Ray" label="On Ray" default>
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
            save_steps=50,
            eval_steps=50,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            max_seq_length=512,
            use_lora=True,
            bits=4,
            ray_address="ray://localhost:10001", # set ray_address
        )
        
        llm = LLM(
            identifier="llm",
            model_name_or_path=model_name,
            trainer=trainer,
            train_X="text", # Which field of data is used for training
            train_select=Collection('datas').find() # Which table data to use for training
        )
        
        llm.fit_in_db(db=db)        
        ```
    </TabItem>
    <TabItem value="Deepspeed" label="Deepspeed" default>
        ```python
        # !pip install deepspeed
        from superduperdb.ext.transformers import LLM, LLMTrainer
        deepspeed = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "zero_optimization": {
                "stage": 2,
            },
        }
        
        trainer = LLMTrainer(
            identifier="llm-finetune-trainer",
            output_dir="output/finetune",
            overwrite_output_dir=True,
            num_train_epochs=3,
            save_total_limit=3,
            logging_steps=10,
            evaluation_strategy="steps",
            save_steps=50,
            eval_steps=50,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            max_seq_length=512,
            use_lora=True,
            bits=4,
            deepspeed=deepspeed, # set deepspped
        )
        
        llm = LLM(
            identifier="llm",
            model_name_or_path=model_name,
            trainer=trainer,
            train_X="text", # Which field of data is used for training
            train_select=Collection('datas').find() # Which table data to use for training
        )
        
        llm.fit_in_db(db=db)        
        ```
    </TabItem>
    <TabItem value="Multi-GPUS" label="Multi-GPUS" default>
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
            save_steps=50,
            eval_steps=50,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            max_seq_length=512,
            use_lora=True,
            bits=4,
            num_gpus=2, # set num_gpus
        )        
        ```
    </TabItem>
</Tabs>
## Load the trained model
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
        checkpoint = db.load("checkpoint", trainer.experiment_id)
        llm = LLM(
            identifier="llm",
            model_name_or_path=model_name,
            adapter_id=checkpoint,
            model_kwargs=dict(load_in_4bit=True)
        )        
        ```
    </TabItem>
</Tabs>
