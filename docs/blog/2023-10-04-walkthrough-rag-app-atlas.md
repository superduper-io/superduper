# The easiest way to implement question-your documents we know

*Despite the huge surge in popularity in building AI applications with LLMs and vector-search,
we haven't seen any walkthroughs boil this down to a super-simple, few-command process.
With SuperDuperDB together with MongoDB Atlas, it's easier and more flexible than ever before.*

---

Setting up a question-your-documents service can be a complex process.

There are several steps involved in doing this:

<!--truncate-->

- Serve a model or forward requests to convert text-data in the database to vectors in a vector-database
- Setting up a vector-index in a vector-database which efficiently finds similar vectors
- Setting up an endpoint to either run a self hosted LLM  or forward requests to a question-answering LLM such as OpenAI
- Setting up an endpoint to:
  - Convert a question to a vector
  - Find relevant documents to the question using vector-search
  - Send the documents as context to the question-answering LLM

This process can be tedious and complex, involving several pieces of infrastructure, especially
if developers would like to use other models than those hosted behind OpenAI's API.

What if we told you that with SuperDuperDB together with MongoDB Atlas, these challenges are a thing of the past, 
and can be done more simply than with any other solution available?

Let's dive straight into the solution:

**Connect to MongoDB Atlas with SuperDuperDB**

```python
from superduperdb.db.base.build import build_datalayer
from superduperdb import CFG
import os

ATLAS_URI = "mongodb+srv://<user>@<atlas-server>/<database_name>"
OPENAI_API_KEY = "<your-open-ai-api-key>"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

CFG.data_backend = ATLAS_URI
CFG.vector_search = ATLAS_URI

db = build_datalayer()
```

After connecting to SuperDuperDB, setting up question-your-documents in Python boils down to 2 commands.

**Set up a vector-index**

```python
from superduperdb.container.vector_index import VectorIndex
from superduperdb.container.listener import Listener
from superduperdb.ext.openai.model import OpenAIEmbedding

collection = Collection('documents')

db.add(
    VectorIndex(
        identifier='my-index',
        indexing_listener=Listener(
            model=OpenAIEmbedding(model='text-embedding-ada-002'),
            key='txt',
            select=collection.find(),
        ),
    )
)
```

In this code snippet, the model used for creating vectors is `OpenAIEmbedding`. This is completely configurable.
You can also use:

- CohereAI API
- Hugging-Face `transformers`
- `sentence-transformers`
- Self built models in `torch`

The `Listener` component sets up this model to listen for new data, and compute new vectors as this data comes in.

The `VectorIndex` connects user queries with the computed vectors and the model.

By adding this nested component to `db`, the components are activated and ready to go for vector-search.

**Add a question-answering component**

```python
from superduperdb.ext.openai.model import OpenAIChatCompletion

chat = OpenAIChatCompletion(
    model='gpt-3.5-turbo',
    prompt=(
        'Use the following content to answer this question\n'
        'Do not use any other information you might have learned\n'
        'Only base your answer on the content provided\n'
        '{context}\n\n'
        'Here\'s the question:\n'
    ),
)

db.add(chat)
```

This command creates and configures an LLM hosted on OpenAI to operate together with MongoDB.
The prompt can be configured to ingest the context using the `{context}` format variable.
The results of the vector search are pasted into this format variable.

**Question your documents!**

```python
input = 'Explain to me the reasons for the change of strategy in the company this year.'

response, context = db.predict(
    'gpt-3.5-turbo',
    input=input,
    context=collection.like({'txt': input}, vector_index='my-index').find()
)
```

This command executes the vector-search query in the `context` parameter. The results of 
this search are added to the prompt to prime the LLM to ground its answer on the documents
in MongoDB.
