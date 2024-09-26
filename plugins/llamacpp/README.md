<!-- Auto-generated content start -->
# superduper_llamacpp

Superduper allows users to work with self-hosted LLM models via [Llama.cpp](https://github.com/ggerganov/llama.cpp).

## Installation

```bash
pip install superduper_llamacpp
```

## API


- [Code](https://github.com/superduper-io/superduper/tree/main/plugins/llamacpp)
- [API-docs](/docs/api/plugins/superduper_llamacpp)

| Class | Description |
|---|---|
| `superduper_llamacpp.model.LlamaCpp` | Llama.cpp connector. |
| `superduper_llamacpp.model.LlamaCppEmbedding` | Llama.cpp connector for embeddings. |


## Examples

### LlamaCpp

```python
from superduper_llama_cpp.model import LlamaCpp

model = LlamaCpp(
    identifier="llm",
    model_name_or_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
)
model.predict("Hello world")
```


<!-- Auto-generated content end -->

<!-- Add your additional content below -->
