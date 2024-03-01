# LLMs
`superduperdb` allows users to work with LLM services and models

Here's the list of supported LLM services/models:

- vLLM
- OpenAI-style API services

## Basic Usage

### Quick start

**Using a model for prediction**

```python
from superduperdb.ext.llm import OpenAI
model = OpenAI(model_name='gpt-3.5-turbo')
model.predict("1+1=")
```

****Using SuperDuperDB to connect ChatGPT with a database for inference****

Create a database layer connection using SuperDuperDB

```python
from superduperdb import superduper
db = superduper("mongomock://test")
```

Insert example datas

```python
from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.document import Document

datas = [Document({"question": f"1+{i}=", "id": str(i)}) for i in range(3)]
db.execute(Collection('docs').insert_many(datas))
```

Add the model to the database

```python
from superduperdb.ext.llm import OpenAI
model = OpenAI(model_name='gpt-4')
db.add(model)
```

Infer the `question` field in the database

```python
model.predict(X='question', db=db, select=Collection('docs').find())
```

### Common Parameter Description

#### **prompt_template**

Used to define the prompt, containing two special fields, default is `"{input}"`

- input: Required, will be filled with the input `"X"`
- context: Optional, the `Context` returned by SuperDuperDB, example usage can be seen in [Building Private Q&A Assistant Using Mongo and Open Source Model](https://github.com/SuperDuperDB/superduperdb/blob/main/examples/question_the_docs_opensource.ipynb)

Example:

```python
from superduperdb.ext.llm import OpenAI
model = OpenAI(model_name='gpt-3.5-turbo', prompt_template="Translate the sentence into Chinese: {input}")
model.predict("Hello")
```

`Hello` will be filled in, and then the LLM model will be called after `Translate the sentence into Chinese: Hello` is obtained.

#### **prompt_func**

A function for custom prompt generation. If provided, this function will be used for prompt construction, and `prompt_template` becomes ineffective.

Example:

```python
from superduperdb.ext.llm import OpenAI
model = OpenAI(model_name='gpt-3.5-turbo', prompt_func=lambda x, lang:  f"Translate the sentence into {lang}: {x}")
model.predict("Hello", lang="Japanese")
```

#### **max_batch_size**

Controls the maximum number of concurrent requests when using API-type LLM models.

#### **predict_kwargs**

All defined inference parameters, which will be sent to the model or API during inference.

```python
from superduperdb.ext.llm import OpenAI
model = OpenAI(model_name='gpt-3.5-turbo', predict_kwargs={'temperature': 0.7})
model.predict("Hello", seed=1)
```

`{"temperature": 0.7, "seed": 1}` will be sent to the OpenAI interface.

Parameters defined in `model.predict` will override those in `predict_kwargs`, but `predict_kwargs` will be registered in the Metadata Store.





## Support Framework/API

### Transformers

[Transformers](https://huggingface.co/docs/transformers/index) is a popular AI framework, and we have incorporated native support for Transformers to provide essential Large Language Model (LLM) capabilities.

You can quickly utilize LLM capabilities using the following Python function:

```python
from superduperdb.ext.llm import LLM
llm = LLM( model_name_or_path="facebook/opt-350m")
llm.predict("What are we having for dinner?")
```

The model can be configured with the following parameters:

- bits: quantization bits, ranging from 4 to 8; the default is None.
- adapter_id: Add an adapter to the base model for inference.
- model_kwargs: a dictionary; all the model_kwargs will be passed to transformers.AutoModelForCausalLM.from_pretrained. You can provide parameters such as trust_remote_code=True.
- tokenizer_kwargs: a dictionary; all the tokenizer_kwargs will be passed to transformers.AutoTokenizer.from_pretrained.
 
### vLLM

[vLLM](https://docs.vllm.ai/en/latest/) is a fast and easy-to-use library for LLM inference and serving.

Currently, `superduperdb` supports the following methods to use vLLM:

- **VllmModel**: Use vLLM to load models.
- **VllmAPI**: Request services in the [API server format](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#api-server).

#### VllmModel

`VllmModel` supports loading models locally or to a Ray cluster.

After instantiation using `model = VllmModel(....)`, the model is loaded lazily and begins loading only when the `model.predict` method is called.

**Load Locally**

To use this model, first install `vllm`

```bash
pip install vllm
```

```python
from superduperdb.ext.llm import VllmModel
model = VllmModel(model_name="mistralai/Mistral-7B-Instruct-v0.2")
```

**Load to a Ray Cluster**

Requires installing `ray`, no need for `vllm` dependencies.

> Installing `vllm` requires a CUDA environment, which can prevent clients without CUDA from installing `vllm`. Therefore, superduperdb has adapted so that if loading to a ray cluster, local installation of `vllm` is not required.

```bash
pip install 'ray[default]'
```

```python
from superduperdb.ext.llm import VllmModel
model = VllmModel(model_name="mistralai/Mistral-7B-Instruct-v0.2", ray_address="ray://ray_cluster_ip:10001")
```

> If this is your first time running on that ray cluster, the wait time might be a bit longer, as `vllm` dependencies and the corresponding model will be installed on the ray cluster's server.

**Parameter**

- model_name: Same as `model` of vLLM

- tensor_parallel_size: Same as `tensor_parallel_size` of vLLM
- trust_remote_code: Same as `trust_remote_code` of vLLM
- vllm_kwargs: Other initialization parameters of vLLM
- on_ray: Whether to run on ray, default False
- ray_address: ray cluster address, if not empty, `on_ray` will automatically be set to True

#### VllmAPI

```python
from superduperdb.ext.llm import VllmAPI
model = VllmAPI(identifier='llm', api_url='http://localhost:8000/v1')
```



#### OpenAI-style vLLM services

If you have deployed [OpenAI-style vLLM services](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server), they can be used with OpenAI as detailed in the following OpenAI format.

### OpenAI

`superduperdb` supports OpenAI-style API services. If parameters like `openai_api_base` are not provided, it defaults to calling OpenAI's services.

**Using a custom API to initialize the model** (example for a vLLM-started OpenAI-style service)

```python
from superduperdb.ext.llm import OpenAI
model = OpenAI(openai_api_base="http://localhost:8000/v1", model_name="mistralai/Mistral-7B-Instruct-v0.2")
```

**Using OpenAI's models**

```python
from superduperdb.ext.llm import OpenAI
model = OpenAI()
```



### Custom Models

If the above models do not meet your needs, you can define your own models as follows, for reference:

#### Non-API Type Models

```python
# https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
import dataclasses as dc
import torch
from transformers import pipeline

from superduperdb.ext.llm import BaseLLMModel

@dc.dataclass
class CustomLLM(BaseLLMModel):
    def init(self):
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def _generate(self, messages, **kwargs):
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
        )
        return outputs[0]["generated_text"]


model = CustomLLM(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```



#### API Type Models

```python
import dataclasses as dc
import os
from openai import OpenAI
from superduperdb.ext.llm import BaseLLMAPI

@dc.dataclass
class CustomModel(BaseLLMAPI):
    def init(self):
        # https://github.com/openai/openai-python#usage
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def _generate(self, prompt: str, **kwargs) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )
        return chat_completion.choices[0].message.content
```



## Fine-tuning

SuperduperDB currently offers convenient support for model fine-tuning.

### Quickly Start

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
print(llm.predict(prompt, max_new_tokens=100, do_sample=True))
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

```
use_lora (`bool`, *optional*, defaults to True):
    Whether to use LoRA training.
    
lora_r (`int`, *optional*, defaults to 8):
    Lora R dimension.

lora_alpha (`int`, *optional*, defaults to 16):
    Lora alpha.

lora_dropout (`float`, *optional*, defaults to 0.05):
    Lora dropout.

lora_target_modules (`List[str]`, *optional*, defaults to None):
    Lora target modules. If None, will be automatically inferred.

lora_bias (`str`, *optional*, defaults to "none"):
    Lora bias.

max_length (`int`, *optional*, defaults to 512):
    Maximum source sequence length during training.
    
log_to_db (`bool`, *optional*, defaults to True):
    Log training to db.
    If True, will log checkpoint to superduperdb,
        but need ray cluster can access to db.
    If can't access to db, please set it to False.
```

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

Provide three Ray-related parameters for configuration:

##### on_ray (str)

Whether to perform training on Ray.

##### ray_address (str)

The address of the Ray cluster to connect to. If not provided, a Ray service will be started locally by default.

##### ray_configs (dict)

All ray_configs will be passed to [TorchTrainer](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html).

Except for the following three fields, which are automatically built by SuperDuperDB:

- train_loop_per_worker
- train_loop_config
- datasets

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

llm.fit(
    X="text",
    select=Collection("datas").find(),
    configuration=base_config,
    db=db,
    on_ray=True,
    ray_address="ray://ray_cluster_ip:10001",
    ray_configs=ray_configs,
)
```

For information on how to configure Ray resources, please refer to the ray documentation, such as:
- [ScalingConfig](https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html#ray.train.ScalingConfig)
- [RunConfig](https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html#ray.train.RunConfig)

