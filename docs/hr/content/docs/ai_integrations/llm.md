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
from superduperdb.components.listener import Listener

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

#### **inference_kwargs**

All defined inference parameters, which will be sent to the model or API during inference.

```python
from superduperdb.ext.llm import OpenAI
model = OpenAI(model_name='gpt-3.5-turbo', inference_kwargs={'temperature': 0.7})
model.predict("Hello", seed=1)
```

`{"temperature": 0.7, "seed": 1}` will be sent to the OpenAI interface.

Parameters defined in `model.predict` will override those in `inference_kwargs`, but `inference_kwargs` will be registered in the Metadata Store.





## Support Framework/API

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



## Training

Coming soon...
