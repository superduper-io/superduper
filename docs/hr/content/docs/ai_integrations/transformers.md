# Transformers

[Transformers](https://huggingface.co/docs/transformers/index) is a popular AI framework, and we have incorporated native support for Transformers to provide essential Large Language Model (LLM) capabilities.
`superduperdb` allows users to work with arbitrary `transformers` pipelines, with custom input/ output data-types.

## Supported `Model` types

### `TextClassification`

...

### `LLM`

You can quickly utilize LLM capabilities using the following Python function:

```python
from superduperdb.ext.transformers import LLM
llm = LLM(model_name_or_path="facebook/opt-350m")
llm.predict_one("What are we having for dinner?")
```

The model can be configured with the following parameters:

- bits: quantization bits, ranging from 4 to 8; the default is None.
- adapter_id: Add an adapter to the base model for inference.
- model_kwargs: a dictionary; all the model_kwargs will be passed to transformers.AutoModelForCausalLM.from_pretrained. You can provide parameters such as trust_remote_code=True.
- tokenizer_kwargs: a dictionary; all the tokenizer_kwargs will be passed to transformers.AutoTokenizer.from_pretrained.

## Training

### LLM fine-tuning

SuperduperDB currently offers convenient support for model fine-tuning.

We can quickly run a fine-tuning example using the qlora finetune Mistral-7B model.

**Install Dependencies**

```bash
pip install transformers torch datasets peft bitsandbytes
```

**Training Script**

```python
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
from superduperdb.base.document import Document
from superduperdb.ext.llm import LLM
from superduperdb.ext.llm.model import LLMTrainingConfiguration

from datasets import load_dataset

model = "mistralai/Mistral-7B-v0.1"
dataset_name = "timdettmers/openassistant-guanaco"

db = superduper("mongomock://test_llm")
dataset = load_dataset(dataset_name)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

train_documents = [
    Document({"text": example["text"], "_fold": "train"})
    for example in train_dataset
]
eval_documents = [
    Document({"text": example["text"], "_fold": "valid"})
    for example in eval_dataset
]

db.execute(Collection("datas").insert_many(train_documents))
db.execute(Collection("datas").insert_many(eval_documents))

trainer = LLMTrainer(
    select=Collection("datas").find(),
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

llm = LLM(
    identifier="llm-finetune",
    bits=4,
    model_name_or_path=model,
    trainer=trainer,
)

db.apply(llm)

prompt = "### Human: Who are you? ### Assistant: "

# Automatically load lora model for prediction, default use the latest checkpoint
print(llm.predict(prompt, max_new_tokens=100, do_sample=True))
```

This script can be found in [`llm_finetune.py`](https://github.com/SuperDuperDB/superduperdb/blob/main/examples/llm_finetune.py).

**Running Training**
You can execute training by running `python examples/llm_finetune.py`.

If you have multiple GPUs, it will automatically use Ray for multi-GPU training.

> If you encounter `ImportError: cannot import name 'ExtensionArrayFormatter' from 'pandas.io.formats.format'` while using multiple GPUs, please downgrade the Pandas version with the following command:
> 
> ```shell
> pip install 'pandas<=2.1.4'
> ```

**Model Usage**
Apart from directly loading and using the model at the end of the script, you can also use your model in other programs provided that you are connected to a real database rather than a mock database.


```python
llm = db.load("model", "llm-finetune")
prompt = "### Human: Who are you? ### Assistant: "
print(llm.predict_one(prompt, max_new_tokens=100, do_sample=True))
```
### Supported Features

**Training Methods**:

- Full fine-tuning
- LoRA fine-tuning

**Parallel Training**:

Parallel training is supported using Ray, with data parallelism as the default strategy. You can also pass DeepSpeed parameters to configure parallelism through [DeepSpeed configuration](https://huggingface.co/docs/transformers/main_classes/deepspeed#zero).

- Multi-GPUs fine-tuning
- Multi-nodes fine-tuning

**Remote Training**:
You can perform remote training by providing a `ray_address`. Imagine you have a Ray cluster with GPUs, you can connect to it from your local machine for training.

### Training Configuration

The training process consists of the following steps:

1. Define a model.
2. Define training parameter configurations.
3. Execute training.

#### Define The Model

```python
llm = LLM(
    identifier="llm-finetune",
    bits=4,
    model_name_or_path=model,
)
```

LLM class model definition can be found in the above introduction.

#### Define Training Parameter Configuration

```python
training_configuration = LLMTrainingConfiguration(
    identifier="llm-finetune-training-config",
    output_dir="output/llm-finetune",
    ...
)
```

The configuration inherits from Huggingface `transformers.TrainingArguments`, which means theoretically you can use any parameters supported by it.

Additionally, some extra parameters are provided to support LLM fine-tuning scenarios.

#### Execute Training

```python
llm.fit(
    X="text",
    select=Collection("datas").find(),
    configuration=training_configuration,
    db=db,
)
```

By default, training will execute directly. However, if multiple GPUs are detected, training will be managed and performed in parallel using Ray.

Additionally, you can manually configure Ray for training, either locally or on a remote Ray cluster.

Provide three `ray`-related parameters for configuration:

##### `on_ray`

Whether to perform training on Ray.

##### `ray_address`

The address of the Ray cluster to connect to. If not provided, a Ray service will be started locally by default.

##### `ray_configs`

All ray_configs will be passed to [ray's `TorchTrainer`](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html).

Except for the following three fields, which are automatically built by SuperDuperDB:

- `train_loop_per_worker`
- `train_loop_config`
- `datasets`

For example, you can provide a configuration like this:

```python
from ray.train import RunConfig, ScalingConfig

scaling_config = ScalingConfig(
    num_workers=4, # Number of GPUs you need
    use_gpu=True,
)

run_config = RunConfig(
    storage_path="s3://llm-test/llm-finetune",
    name="llm-finetune-test100",
)

ray_configs = {
    "scaling_config": scaling_config,
    "run_config": run_config,
}

trainer = LLMTrainer(
    ...,
    ray_configs=ray_configs,
    on_ray=True
)

db.apply(
    LLM(
        ...,
        trainer=trainer,
    )
)
```

For information on how to configure Ray resources, please refer to the ray documentation, such as:
- [ScalingConfig](https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html#ray.train.ScalingConfig)
- [RunConfig](https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html#ray.train.RunConfig)
