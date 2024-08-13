---
sidebar_position: 3
---

# PyTorch

`Superduper` allows users to work with arbitrary `torch` models, with custom pre-, post-processing and input/ output data-types,
as well as offering training with `Superduper`


| Class | Description | GitHub | API-docs |
| --- | --- | --- | --- |
| `superduper.ext.torch.model.TorchModel` | Wraps a PyTorch model | [Code](https://github.com/superduper/superduper/blob/main/superduper/ext/torch/model.py) | [Docs](/docs/api/ext/torch/model#torchmodel-1) |
| `superduper.ext.torch.model.TorchTrainer` | May be attached to a `TorchModel` for training | [Code](https://github.com/superduper/superduper/blob/main/superduper/ext/torch/training.py) | [Docs](/docs/api/ext/torch/training#torchtrainer)