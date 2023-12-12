---
sidebar_position: 19
---

# Using AI APIs as `Predictor` descendants

In SuperDuperDB, developers are able to interact with popular AI API providers, in a way very similar to 
[integrating with AI open-source or home-grown models](./ai_models.md). Instantiating a model from 
these providers is similar to instantiating a `Model`:

## OpenAI

**Supported**

| Description | Class-name |
| --- | --- |
| Embeddings | `OpenAIEmbedding` |
| Chat models | `OpenAIChatCompletion` |
| Image generation models | `OpenAIImageCreation` |
| Image edit models | `OpenAIImageEdit` |
| Audio transcription models | `OpenAIAudioTranscription` |

**Usage**

```python
from superduperdb.ext.openai import OpenAI<ModelType> as ModelCls

db.add(Modelcls(identifier='my-model', **kwargs))
```

## Cohere

**Supported**

| Description | Class-name |
| --- | --- |
| Embeddings | `CohereEmbedding` |
| Chat models | `CohereChatCompletion` |

**Usage**

```python
from superduperdb.ext.cohere import Cohere<ModelType> as ModelCls

db.add(Modelcls(identifier='my-model', **kwargs))
```

## Anthropic

**Supported**

| Description | Class-name |
| --- | --- |
| Chat models | `AnthropicCompletions` |

**Usage**

```python
from superduperdb.ext.anthropic import Anthropic<ModelType> as ModelCls

db.add(Modelcls(identifier='my-model', **kwargs))
```

## Jina

**Supported**

| Description | Class-name |
| --- | --- |
| Embeddings | `JinaEmbedding` |

**Usage**

```python
from superduperdb.ext.jina import JinaEmbedding

db.add(JinaEmbedding(identifier='jina-embeddings-v2-base-en', api_key='JINA_API_KEY')) # You can also set JINA_API_KEY as environment variable
```