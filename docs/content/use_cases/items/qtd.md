# Building a RAG application with FastAPI, MongoDB and SuperDuperDB

In this use-case, we'll use FastAPI to serve SuperDuperDB on fly.io using MongoDB as a databackend.
The task we'll be implementing is a retrieval augmented text-generation (RAG) app for answering
questions about a particular trove of documents. Read [on our blog](/blog/...) for more background.

## Create the FastAPI app file structure

There are many choices here. Please refer to the `FastAPI` documentation for other possible choices. The structure that we chose looks like the following:

```python
backend
├── ai  # RAG app-specific code
│   ├── ...
├── documents  # routes for our app
│   ├── __init__.py
│   ├── models.py  # pydantic models
│   └── routes.py  # AI-enhanced CRUD logic
├── __init__.py
├── app.py  # events that occur at app startup/shutdown
├── config.py
└── main.py
```

## Add logic for events on startup and shutdown of the app

As we are working with a CRUD-like app, we want to establish a connection to the database on app startup. We also want to close this connection on app shutdown. We use `FastAPI` event handlers to perform this logic. 

```python
# app.py

from superduperdb import superduper

...

@_app.on_event('startup')
def startup_db_client():
    _app.mongodb_client = MongoClient(settings.mongo_uri)
    _app.mongodb = _app.mongodb_client[settings.mongo_db_name]

    _app.superduperdb = superduper(_app.mongodb)
    ...

@_app.on_event('shutdown')
def shutdown_db_client():
    _app.mongodb_client.close()
```

It is important that the database connection is wrapped with the `superduper` function. This is how the underlying database driver (`pymongo` for our RAG app) becomes 'superduperdb-aware'.

Two other events that happen on startup of the application are:
1. load the AI models (`components`) into the MongoDB database
2. load the documentation into the database in a suitable format for vector-similarity search (`artifacts`)

```python
# app.py

...

load_ai_artifacts(_app.superduperdb)
install_ai_components(_app.superduperdb)
```

See below for more details on both `components` and `artifacts`.

The final action that we perform on app startup is to initialise our routes. This is a common pattern in `FastAPI` apps:

```python
# app.py

def init_routers(app: FastAPI) -> None:
    app.include_router(documents_router)

 def create_app() -> FastAPI:
    _app = FastAPI(title='Question the Docs')   

    ...

    init_routers(app=_app)
```

## Load the AI models (`components`) into the MongoDB database

A RAG app is built from combining an LLM with some means of performing vector-similarity search on real-time data. In our RAG app, we use the OpenAI `ChatGPT` model as our LLM. This functionality is available out of the box with `superduperdb`. To install this LLM in our app, we add it to our database as follows:

```python
# ai/components.py

from superduperdb.ext.openai.model import OpenAIChatCompletion

def install_openai_chatbot(db):
    db.add(
        OpenAIChatCompletion(
            takes_context=True,
            prompt=settings.prompt,
            model=settings.qa_model,
        )
    )
```

`db` is the SuperDuperDB-wrapped database object from [step 2](#step-2-add-logic-for-events-on-startup-and-shutdown-of-the-app). The `settings` module contains global configuration options for the app, including the version of `ChatGPT` to use and the prompt template to pass to `ChatGPT` for conversation.

To perform vector-similarity search, we use the `VectorIndex` object from SuperDuperDB, in association with embeddings from the OpenAI API:

```python
# ai/components.py

from superduperdb.container.listener import Listener
from superduperdb.container.vector_index import VectorIndex
from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.openai.model import OpenAIEmbedding

def install_openai_vector_index(db, repo):
    db.add(
        VectorIndex(
            identifier=repo,
            indexing_listener=Listener(
                model=OpenAIEmbedding(model=settings.vector_embedding_model),
                key=settings.vector_embedding_key,
                select=Collection(name=repo).find(),
                predict_kwargs={'chunk_size': 100},
            ),
        )
    )
```

`VectorIndex` is a SuperDuperDB abstraction that is used for adding vector-search functionality. It has support for a range of vector-database options, including `lance` which is a fully-embedded vector database option.

A `Listener` is another SuperDuperDB abstraction that is used to 'listen' for changes to data in the underlying datastore (in our RAG app, this is `MongoDB`). When changes are detected, it will execute a callback function on the data. In this case, our callback is a call to the OpenAI API to recompute the vector embeddings of the data.

The remaining keyword arguments all control the behaviour of our vector-search and 'change-data-capture' functionality. For example, the `chunk_size` keyword controls how many items to recompute in each batch.

## Load the documentation into MongoDB

Before performing vector-similarity searches on pieces of text, we need to first create vector representations of the text. Ideally, we want similar pieces of text to have similar vector representations. 

The first part of this challenge involves using a suitable Embedding model. Here, we use the OpenAI `text-embedding-ada-002` model. The second part of the challenge involves deciding on the length of each piece of text to vectorise. For example, for technical documentation we might decide that each paragraph represents a unit of information, and so we should vectorise each paragraph. There is no 'correct' answer here, and it will depend to some extent on the application that is being built. SuperDuperDB supports a range of tools out-of-the-box to help with these tasks such as `spacy`, `torchvision` and `transformers`.

Once these embeddings have been created, they need to be saved to the database. In our app, this looks like the following:

```python
# ai/artifacts.py

from superduperdb.container.document import Document
from superduperdb.db.mongodb.query import Collection

def load_ai_artifacts(db):
    ...

    documents = [
        Document(
            {settings.vector_embedding_key: row['text'], 'src_url': row['src_url']}
            )
        for _, row in artifacts.iterrows()
    ]
    db.execute(Collection(name=repo).insert_many(documents))
```

`Document` is a convenience wrapper around items that we wish to store in our database. Here, we see that each item is a dictionary that consists of two key-value pairs. The first key-value pair represents a piece of text and its vector representation. The second key-value pair represents the location ('URL') of this piece of text in the documentation. Providing sources alongside each answer is a practical strategy for dealing with LLM hallucinations.

Finally, we insert all this information into our MongoDB database by 'executing' the `insert_many` command, which should be familiar to MongoDB users.

## Building our routes

Every `FastAPI` will consist of a series of endpoints, or routes. In our RAG app we have two routes. The first route performs a vector-similarity search on a piece of text. A user submits a question, and a collection of texts from the technical documentation that are most similar to the user query will be returned:

```python
# documents/routes.py

from backend.documents.models import Query, Source

from superduperdb.db.mongodb.query import Collection

...

async def vector_search(request: Request, query: Query) -> Source:
    collection = Collection(name=query.collection_name)
    context_select = collection.like(
        {settings.vector_embedding_key: query.query},
        n=settings.nearest_to_query,
        vector_index=query.collection_name,
    ).find()

    db = request.app.superduperdb
    contexts = list(db.execute(context_select))
    ...
```

There are two parts to this route. The first part builds the query. It users an API that is very similar to the MongoDB query API. The second part executes the query, and loads it into memory as a `list` of texts.

The second endpoint does something similar, but it also formats a prompt with the results of the vector-similarity search, and then sends this prompt to `ChatGPT` for summarization. Because LLMs can take some seconds to respond, we interact with `ChatGPT` in 'streaming' mode. This gives a feel of near real-time response to the app:

```python
# documents/routes.py

async def query_docs(request: Request, query: Query) -> StreamingResponse:
    ...

    db = request.app.superduperdb
    db_response, _ = await db.apredict(
        'gpt-3.5-turbo',
        input=query.query,
        context_select=context_select,
        context_key=settings.vector_embedding_key,
        stream=True,
    )
    ...
```

## Deploy the app

At this point, the backend to our app is ready to be deployed. There are a range of options available here. The option that we chose is [fly.io](https://fly.io/). Check out [the application code](https://github.com/SuperDuperDB/superduperdb/tree/main/apps/question-the-docs) in our main repo to see exactly how everything is configured!