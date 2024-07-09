---
sidebar_position: 18
---

# Applying `Model` instances to `db`

There are 4 key AI `Model` sub classes, see [here](../apply_api/model) for detailed usage:

| Path | Description |
| --- | ---
| `superduper.components.model.ObjectModel` | Wraps a Python object to compute outputs |
| `superduper.components.model.APIModel` | Wraps a model hosted behind an API to compute outputs |
| `superduper.components.model.QueryModel` | Maps a Database select query with a free variable over inputs |
| `superduper.components.model.SequentialModel` | Computes outputs sequentially for a sequence of `Model` instances |

As well as these key sub-classes, we have classes in the `superduper.ext.*` subpackages:
See [here](../ai_integrations/) for more information.

Whenever one of these `Model` descendants is instantiated, and `db.apply(model)` is called, 
several things can (do) happen:

1. The `Model`'s metadata is saved in the `db.metadata_store`.
2. It's associated data (e.g.) model is saved in the `db.artifact_store`.
3. (Optional) if the `Model` has a `Trainer` attached, then the `Model` is trained/ fit on the specified data.
4. (Optional) if the `Model` has an `Evaluation` method attached, then the `Model` is evaluated on the specified data.

<!-- ### Scikit-Learn

```python
from superduper.ext.sklearn import Estimator
from sklearn.svm import SVC

db.add(Estimator(SVC()))
```

### Transformers

```pytho
from superduper.ext.transformers import Pipeline
from superduper import superduper

db.add(Pipeline(task='sentiment-analysis'))
```

There is also support for building the pipeline in separate stages with a high degree of customization.
The following is a speech-to-text model published by [facebook research](https://arxiv.org/abs/2010.05171) and shared [on Hugging-Face](https://huggingface.co/facebook/s2t-small-librispeech-asr):

```python
from superduper.ext.transformers import Pipeline
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

transcriber = Pipeline(
    identifier='transcription',
    object=model,
    preprocess=processor,
    preprocess_kwargs={'sampling_rate': SAMPLING_RATE, 'return_tensors': 'pt', 'padding': True}, # Please replace the placeholder `SAMPLING_RATE` with the appropriate value in your context.
    postprocess=lambda x: processor.batch_decode(x, skip_special_tokens=True),
    predict_method='generate',
    preprocess_type='other',
)

db.add(transcriber)
```

### PyTorch

```python
import torch
from superduper.ext.torch import Module

model = Module(
    identifier='my-classifier',
    preprocess=lambda x: torch.tensor(x),
    object=torch.nn.Linear(64, 512),
    postprocess=lambda x: x.topk(1)[0].item(),
)

db.add(model)
```

### Important Parameters, Common to All Models
  
| Name | Function |
| --- | --- |
| `identifier` | A unique name for `superduper`, for later use and recall |
| `object` | The model-object, including parameters and hyper-parameters providing heavy lifting |
| `preprocess` | `Callable` which processes individual rows/records/fields from the database prior to passing to the model |
| `postprocess` | `Callable` applied to individual rows/items or output |
| `encoder` | An `Encoder` instance applied to the model output to save that output in the database |
| `schema` | A `Schema` instance applied to a model's output, whose rows are dictionaries |


## Using AI APIs 

In superduper, developers are able to interact with popular AI API providers, in a way very similar to 
[integrating with AI open-source or home-grown models](./ai_models.md). Instantiating a model from 
these providers is similar to instantiating a `Model`:

### OpenAI

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
from superduper.ext.openai import OpenAI<ModelType> as ModelCls

db.add(Modelcls(identifier='my-model', **kwargs))
```

### Cohere

**Supported**

| Description | Class-name |
| --- | --- |
| Embeddings | `CohereEmbedding` |
| Chat models | `CohereChatCompletion` |

**Usage**

```python
from superduper.ext.cohere import Cohere<ModelType> as ModelCls

db.add(Modelcls(identifier='my-model', **kwargs))
```

### Anthropic

**Supported**

| Description | Class-name |
| --- | --- |
| Chat models | `AnthropicCompletions` |

**Usage**

```python
from superduper.ext.anthropic import Anthropic<ModelType> as ModelCls

db.add(Modelcls(identifier='my-model', **kwargs))
```

### Jina

**Supported**

| Description | Class-name |
| --- | --- |
| Embeddings | `JinaEmbedding` |

**Usage**

```python
from superduper.ext.jina import JinaEmbedding

db.add(JinaEmbedding(identifier='jina-embeddings-v2-base-en', api_key='JINA_API_KEY')) # You can also set JINA_API_KEY as environment variable
``` -->