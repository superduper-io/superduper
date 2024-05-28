---
sidebar_position: 3
---

# PyTorch

`superduperdb` allows users to work with arbitrary `torch` models, with custom pre-, post-processing and input/ output data-types,
as well as offering training with `superduperdb`


| Class | Description | GitHub | API-docs |
| --- | --- | --- | --- |
| `superduperdb.ext.torch.model.TorchModel` | Wraps a PyTorch model | [Code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/ext/torch/model.py) | [Docs](/docs/api/ext/torch/model#torchmodel-1) |
| `superduperdb.ext.torch.model.TorchTrainer` | May be attached to a `TorchModel` for training | [Code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/ext/torch/training.py) | [Docs](/docs/api/ext/torch/training#torchtrainer)