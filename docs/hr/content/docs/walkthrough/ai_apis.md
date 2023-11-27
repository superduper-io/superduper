---
sidebar_position: 19
---

# Using AI APIs as `Predictor` descendants

In SuperDuperDB, developers are able to interact with popular AI API providers, in a way very similar to 
[integrating with AI open-source or home-grown models](18_ai_models.mdx). Instantiating a model from 
these providers is similar to instantiating a `Model`:

## OpenAI

Supported:

- Embeddings
- Chat models
- Image generation models
- Image edit models
- Audio transcription models

The general pattern is:

```python
from superduperdb.ext.openai import OpenAI<ModelType> as ModelCls

db.add(Modelcls(identifier='my-model', **kwargs))
```

## Cohere

```python
from superduperdb.ext.cohere import Cohere<ModelType> as ModelCls

db.add(Modelcls(identifier='my-model', **kwargs))
```

## Anthropic

```python
from superduperdb.ext.anthropic import Anthropic<ModelType> as ModelCls

db.add(Modelcls(identifier='my-model', **kwargs))
```