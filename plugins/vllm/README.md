<!-- Auto-generated content start -->
# superduper_vllm

Superduper allows users to work with self-hosted LLM models via [vLLM](https://github.com/vllm-project/vllm).

## Installation

```bash
pip install superduper_vllm
```

## API


- [Code](https://github.com/superduper-io/superduper/tree/main/plugins/vllm)
- [API-docs](/docs/api/plugins/superduper_vllm)

| Class | Description |
|---|---|
| `superduper_vllm.model.VllmChat` | VLLM model for chatting. |
| `superduper_vllm.model.VllmCompletion` | VLLM model for generating completions. |


## Examples

### VllmChat

```python
from superduper_vllm import VllmChat
vllm_params = dict(
    model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    quantization="awq",
    dtype="auto",
    max_model_len=1024,
    tensor_parallel_size=1,
)
model = VllmChat(identifier="model", vllm_params=vllm_params)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hello"},
]
```

Chat with chat format messages

```python
model.predict(messages)
```

Chat with text format messages

```python
model.predict("hello")
```

### VllmCompletion

```python
from superduper_vllm import VllmCompletion
vllm_params = dict(
    model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    quantization="awq",
    dtype="auto",
    max_model_len=1024,
    tensor_parallel_size=1,
)
model = VllmCompletion(identifier="model", vllm_params=vllm_params)
model.predict("hello")
```


<!-- Auto-generated content end -->

<!-- Add your additional content below -->
