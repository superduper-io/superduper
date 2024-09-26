<!-- Auto-generated content start -->
# superduper_transformers

Transformers is a popular AI framework, and we have incorporated native support for Transformers to provide essential Large Language Model (LLM) capabilities.

Superduper allows users to work with arbitrary transformers pipelines, with custom input/ output data-types.


## Installation

```bash
pip install superduper_transformers
```

## API


- [Code](https://github.com/superduper-io/superduper/tree/main/plugins/transformers)
- [API-docs](/docs/api/plugins/superduper_transformers)

| Class | Description |
|---|---|
| `superduper_transformers.model.TextClassificationPipeline` | A wrapper for ``transformers.Pipeline``. |
| `superduper_transformers.model.LLM` | LLM model based on `transformers` library. |


## Examples

### TextClassificationPipeline

```python
from superduper_transformers.model import TextClassificationPipeline

model = TextClassificationPipeline(
    identifier="my-sentiment-analysis",
    model_name="distilbert-base-uncased",
    model_kwargs={"num_labels": 2},
    device="cpu",
)
model.predict("Hello, world!")
```

### LLM

```python
from superduper_transformers import LLM
model = LLM(identifier="llm", model_name_or_path="facebook/opt-125m")
model.predict("Hello, world!")
```


<!-- Auto-generated content end -->

<!-- Add your additional content below -->
## Training Example
Read more about this [here](https://docs.superduper.io/docs/templates/llm_finetuning)
