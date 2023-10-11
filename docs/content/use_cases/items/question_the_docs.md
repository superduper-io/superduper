# Ask the docs anything about SuperDuperDB

In this notebook we show you how to implement the much-loved document Q&A task, using SuperDuperDB
together with MongoDB.


```python
!pip install superduperdb==0.0.12
```


```python
import os
os.environ['OPENAI_API_KEY'] = '<YOUR-OPENAI-API-KEY>'
```


```python
import os
from superduperdb import superduper
from superduperdb.db.mongodb.query import Collection

# Uncomment one of the following lines to use a bespoke MongoDB deployment
# For testing the default connection is to mongomock

mongodb_uri = os.getenv("MONGODB_URI","mongomock://test")
# mongodb_uri = "mongodb://localhost:27017"
# mongodb_uri = "mongodb://superduper:superduper@mongodb:27017/documents"
# mongodb_uri = "mongodb://<user>:<pass>@<mongo_cluster>/<database>"
# mongodb_uri = "mongodb+srv://<username>:<password>@<atlas_cluster>/<database>"

# Super-Duper your Database!
from superduperdb import superduper
db = superduper(mongodb_uri)

collection = Collection('questiondocs')
```

In this example we use the internal textual data from the `superduperdb` project's API documentation, with the "meta"-goal of 
creating a chat-bot to tell us about the project which we are using!


```python
import glob
glob.glob('../superduperdb/docs/content/docs/*/*.md')
```


```python
import glob

STRIDE = 5       # stride in numbers of lines
WINDOW = 10       # length of window in numbers of lines

content = sum([open(file).readlines() 
               for file in glob.glob('../superduperdb/docs/content/docs/*/*.md') 
               + glob.glob('../superduperdb/docs/content/docs/*.md')], [])
chunks = ['\n'.join(content[i: i + WINDOW]) for i in range(0, len(content), STRIDE)]
```

You can see that the chunks of text contain bits of code, and explanations, 
which can become useful in building a document Q&A chatbot.


```python
from IPython.display import Markdown
Markdown(chunks[1])
```

As usual we insert the data:


```python
from superduperdb.container.document import Document

db.execute(collection.insert_many([Document({'txt': chunk}) for chunk in chunks]))
```

We set up a standard `superduperdb` vector-search index using `openai` (although there are many options
here: `torch`, `sentence_transformers`, `transformers`, ...)


```python
from superduperdb.container.vector_index import VectorIndex
from superduperdb.container.listener import Listener
from superduperdb.ext.openai.model import OpenAIEmbedding

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

Now we create a chat-completion component, and add this to the system:


```python
from superduperdb.ext.openai.model import OpenAIChatCompletion

chat = OpenAIChatCompletion(
    model='gpt-3.5-turbo',
    prompt=(
        'Use the following description and code-snippets aboout SuperDuperDB to answer this question about SuperDuperDB\n'
        'Do not use any other information you might have learned about other python packages\n'
        'Only base your answer on the code-snippets retrieved\n'
        '{context}\n\n'
        'Here\'s the question:\n'
    ),
)

db.add(chat)
```

We can view that this is now registed in the system:


```python
print(db.show('model'))
```

Finally, asking questions about the documents can be targeted with a particular query.
Using the power of MongoDB, this allows users to use vector-search in combination with
important filtering rules:


```python
from superduperdb.container.document import Document
from IPython.display import display, Markdown

q = 'Can you give me a code-snippet to set up a `VectorIndex`?'

output, context = db.predict(
    model_name='gpt-3.5-turbo',
    input=q,
    context_select=(
        collection
            .like(Document({'txt': q}), vector_index='my-index', n=5)
            .find()
    ),
    context_key='txt',
)

Markdown(output.content)
```
