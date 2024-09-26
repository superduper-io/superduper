<!-- Auto-generated content start -->
# superduper_torch

Superduper allows users to work with arbitrary `torch` models, with custom pre-, post-processing and input/ output data-types, as well as offering training with superduper

## Installation

```bash
pip install superduper_torch
```

## API


- [Code](https://github.com/superduper-io/superduper/tree/main/plugins/torch)
- [API-docs](/docs/api/plugins/superduper_torch)

| Class | Description |
|---|---|
| `superduper_torch.model.TorchModel` | Torch model. This class is a wrapper around a PyTorch model. |
| `superduper_torch.training.TorchTrainer` | Configuration for the PyTorch trainer. |


## Examples

### TorchModel

```python
import torch
from superduper_torch.model import TorchModel

model = TorchModel(
    object=torch.nn.Linear(32, 1),
    identifier="test",
    preferred_devices=("cpu",),
    postprocess=lambda x: int(torch.sigmoid(x).item() > 0.5),
)
model.predict(torch.randn(32))
```


<!-- Auto-generated content end -->

<!-- Add your additional content below -->

## Training Example
Read more about this [here](https://docs.superduper.io/docs/templates/transfer_learning)
