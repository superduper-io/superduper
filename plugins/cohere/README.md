<!-- Auto-generated content start -->
# superduper_cohere

Superduper allows users to work with cohere API models.

## Installation

```bash
pip install superduper_cohere
```

## API


- [Code](https://github.com/superduper-io/superduper/tree/main/plugins/cohere)
- [API-docs](/docs/api/plugins/superduper_cohere)

| Class | Description |
|---|---|
| `superduper_cohere.model.Cohere` | Cohere predictor. |
| `superduper_cohere.model.CohereEmbed` | Cohere embedding predictor. |
| `superduper_cohere.model.CohereGenerate` | Cohere realistic text generator (chat predictor). |


## Examples

### CohereEmbed

```python
from superduper_cohere.model import CohereEmbed
model = CohereEmbed(identifier='embed-english-v2.0', batch_size=1)
model..predict('Hello world')
```

### CohereGenerate

```python
from superduper_cohere.model import CohereGenerate
model = CohereGenerate(identifier='base-light', prompt='Hello, {context}')
model.predict('', context=['world!'])
```


<!-- Auto-generated content end -->

<!-- Add your additional content below -->
