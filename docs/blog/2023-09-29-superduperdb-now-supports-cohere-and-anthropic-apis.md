# SuperDuperDB now supports Cohere and Anthropic APIs

*We're happy to announce the integration of two more AI APIs, Cohere and Anthropic, into SuperDuperDB.*

---

[Cohere](https://cohere.com/) and [Anthropic](https://www.anthropic.com/) provides AI developers with sorely needed alternatives to OpenAI for key AI tasks, 
including:

- text-embeddings as a service
- chat-completions as as service.

<!--truncate-->

## How to use the new integrations with your data

We've posted some extensive content on how to use [OpenAI or Sentence-Transformers for encoding text as vectors](https://docs.superduperdb.com/docs/use_cases/items/compare_vector_search_solutions)
in your MongoDB datastore, and also [how to question-your your docs hosted in MongoDB](https://docs.superduperdb.com/blog/building-a-documentation-chatbot-using-fastapi-react-mongodb-and-superduperdb).

Now developers can easily use Cohere and Anthropic and drop in-replacements for the OpenAI models:

For **embedding text as vectors** with OpenAI developers can continue to import the functionality like this:

```python
from superduperdb.ext.openai.model import OpenAIEmbedding

model = OpenAIEmbedding(model='text-embedding-ada-002')
```

Now developers can **also** import Cohere's embedding functionality:

```python
from superduperdb.ext.cohere.model import CohereEmbedding

model = CohereEmbedding()
```

Similarly, for **chat-completion**, can continue to use OpenAI like this:

```python
from superduperdb.ext.openai.model import OpenAIChatCompletion

chat = OpenAIChatCompletion(
    prompt='Answer the following question clearly, concisely and accurately',
    model='gpt-3.5-turbo',
)
```

Now developers can **also** import Anthropic's embedding functionality:

```python
from superduperdb.ext.anthropic.model import AnthropicCompletions

chat = AnthropicCompletions(
    prompt='Answer the following question clearly, concisely and accurately',
)
```
