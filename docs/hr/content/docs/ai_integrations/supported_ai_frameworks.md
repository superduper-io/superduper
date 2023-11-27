---
sidebar_position: 1
---

# Community Support

The primary way in which developers will integrate and implement functionality from popular AI frameworks, is via
the `Predictor` and `Model` abstractions.

The `Predictor` mixin class, interfaces with all AI frameworks and API providers, which provide `self.predict` functionality,
and is subclassed by:

| class | framework |
| --- | --- |
| `superduperdb.ext.sklearn.Estimator` | [Scikit-Learn](https://scikit-learn.org/stable/) |
| `superduperdb.ext.transformers.Pipeline` | [Hugging Face's `transformers`](https://huggingface.co/docs/transformers/index) |
| `superduperdb.ext.torch.TorchModel` | [PyTorch](https://pytorch.org/) |
| `superduperdb.ext.openai.OpenAI` | [OpenAI](https://api.openai.com) |
| `superduperdb.ext.cohere.Cohere` | [Cohere](https://cohere.com) |
| `superduperdb.ext.anthropic.Anthropic` | [Anthropic](https://anthropic.com) |

The `Model` class is subclassed by:

| class | framework |
| --- | --- |
| `superduperdb.ext.sklearn.Estimator` | [Scikit-Learn](https://scikit-learn.org/stable/) |
| `superduperdb.ext.transformers.Pipeline` | [Hugging Face's `transformers`](https://huggingface.co/docs/transformers/index) |
| `superduperdb.ext.torch.TorchModel` | [PyTorch](https://pytorch.org/) |

`Model` instances implement `self.predict`, but also hold import data, such as model weights, parameters or hyperparameters.
In addition, `Model` may implement `self.fit` functionality - model training and calibration.