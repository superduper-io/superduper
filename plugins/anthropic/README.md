<!-- Auto-generated content start -->
# superduper_anthropic

Superduper allows users to work with anthropic API models. The key integration is the integration of high-quality API-hosted LLM services.

## Installation

```bash
pip install superduper_anthropic
```

## API


- [Code](https://github.com/superduper-io/superduper/tree/main/plugins/anthropic)
- [API-docs](/docs/api/plugins/superduper_anthropic)

| Class | Description |
|---|---|
| `superduper_anthropic.model.Anthropic` | Anthropic predictor. |
| `superduper_anthropic.model.AnthropicCompletions` | Cohere completions (chat) predictor. |


## Examples

### AnthropicCompletions

```python
from superduper_anthropic.model import AnthropicCompletions

model = AnthropicCompletions(
    identifier="claude-2.1",
    predict_kwargs={"max_tokens": 64},
)
model.predict_batches(["Hello, world!"])
```


<!-- Auto-generated content end -->

<!-- Add your additional content below -->
