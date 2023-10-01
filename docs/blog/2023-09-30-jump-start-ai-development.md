# Jumpstart AI development on MongoDB with SuperDuperDB



MongoDB now supports vector-search on Atlas! With vector-search developers can easily build these types of AI applications:

- **semantic-search** (search by meaning)
- **chat applications** on documents hosted in MongoDB Atlas clusters

SuperDuperDB is an open-source (Apache 2.0 license) python project enabling users to bring AI to their datastore with as little pain as possible. This includes first-class support for MongoDB.

<!--truncate-->

## Using SuperDuperDB to get started with Atlas vector-search

There is great content on the MongoDB website on how to [get started with vector-search on Atlas](https://www.mongodb.com/library/vector-search/building-generative-ai-applications-using-mongodb). You'll see that there are several steps involved:

1. Preparing documents for vector-search
2. Converting text into vectors with an AI "model" and storing these vectors in MongoDB
3. Setting up a vector-search index on Atlas vector-search
4. Preparing a production API endpoint to convert searches in real time to vectors

Each of these steps contains several sub-steps, and can become quite a headache for developers wanting to get started with vector-search.

With SuperDuperDB, this preparation process can be boiled down to one simple command:

```python
from superduperdb.ext.openai.model import OpenAIEmbedding
from superduperdb.container.vector_index import VectorIndex
from superduperdb.container.listener import Listener
from superduperdb.db.mongodb.query import Collection

db.add(
    VectorIndex(
        identifier='my-index',
        indexing_listener=Listener(
            model=OpenAIEmbedding(model='text-embedding-ada-002'),
            key='key',  # path of documents
            select=Collection('documents').find(),
            predict_kwargs={'max_chunk_size': 1000},
        ),
    )
)
```

Under the hood SuperDuperDB does these things:

1. Sets up an Atlas vector-search index in the `"documents"` collection
2. Converts all documents into vectors
3. Creates a function allow users to directly search using vectors, without needing to handle the conversion to vectors themselves: `Collection('documents').like({'key': 'This is the text to search with'}).find()`. This function can easily be served using, for example, FastAPI. (See [here](https://docs.superduperdb.com/blog/building-a-documentation-chatbot-using-fastapi-react-mongodb-and-superduperdb) for an example.)

## Take AI even further with SuperDuperDB on MongoDB

AI is not just vector-search over text-documents -- there are countless additional ways in which AI can be leveraged with data. This is where SuperDuperDB excels and other solutions come up short in leveraging data in MongoDB. 

SuperDuperDB also allows developers to:

- Search the content of [images](https://docs.superduperdb.com/docs/use_cases/items/multimodal_image_search_clip), videos and [voice memos](https://docs.superduperdb.com/docs/use_cases/items/voice_memos) in MongoDB
- Create [talk-to-your documents style chat applications](https://docs.superduperdb.com/blog/building-a-documentation-chatbot-using-fastapi-react-mongodb-and-superduperdb).
- Use classical machine learning models [together with state-of-the-art computer vision models](https://docs.superduperdb.com/docs/use_cases/items/resnet_features). 