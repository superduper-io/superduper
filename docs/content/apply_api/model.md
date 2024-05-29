# `Model`

- Wrap a standard AI model with functionality necessary for SuperDuperDB
- Configure validation and training of a model on database data

***Dependencies***

- [`Datatype`](./datatype.md)

***(Optional dependencies)***

- [`Validation`](./validation.md)
- [`Trainer`](./trainer.md)

***Usage pattern***

:::note
Note that `Model` is an abstract base class which cannot be called directly.
To use `Model` you should call any of its downstream implementations, 
such as [`ObjectModel`](../api/components/model.md#objectmodel-1) or models in the [AI-integrations](/docs/category/ai-integrations).
:::

***Important notes***

`Model` instances can output data not-usually supported by your database.
This data will be encoded by default by `pickle`, but more control may be added
by adding the parameters `datatype=...` or `output_schema=...`.

## Implementations

Here are a few SuperDuperDB native implementations:

**`ObjectModel`**

Use a self-built model (`object`) or function with the system:

```python
from superduperdb import ObjectModel

m = ObjectModel(
    'my-model',
    object=lambda x: x + 2,
)

db.apply(m)
```

**`QueryModel`**

Use a SuperDuperDB query to extract data from `db`

```python
from superduperdb.components.model import QueryModel

query = ... # build a select query
m = QueryModel('my-query', select=query, key='<key-to-extract>')

db.apply(m)
```

**`APIModel`**

Request model outputs hosted behind an API:

```python
from superduperdb.components.model import APIModel

m = APIModel('my-api', url='http://localhost:6666?token={MY_DEV_TOKEN}&model={model}&text={text}')

db.apply(m)
```

**`SequentialModel`**

Make predictions on the basis of a sequence of models:

```python
from superduperdb.components.model import SequentialModel

m = SequentialModel(
    'my-sequence',
    models=[
        model1,
        model2,
        model3,
    ]
)

db.apply(m)
```

***See also***

- [Scikit-learn extension](../ai_integrations/sklearn)
- [Pytorch extension](../ai_integrations/pytorch)
- [Transformers extension](../ai_integrations/transformers)
- [Llama.cpp extension](../ai_integrations/llama_cpp)
- [Vllm extension](../ai_integrations/vllm)
- [OpenAI extension](../ai_integrations/openai)
- [Anthropic extension](../ai_integrations/anthropic)
- [Cohere extension](../ai_integrations/cohere)
- [Jina extension](../ai_integrations/jina)