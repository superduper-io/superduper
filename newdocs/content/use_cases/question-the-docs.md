# Ask the docs anything about SuperDuperDB


```python
import os
os.environ['OPENAI_API_KEY'] = '<YOUR-OPENAI-API-KEY>'
```


```python
from superduperdb import superduper
from superduperdb.db.mongodb.query import Collection
import pymongo

db = pymongo.MongoClient().documents
db = superduper(db)

collection = Collection('questiondocs')
```


```python
import glob

STRIDE = 5       # stride in numbers of lines
WINDOW = 10       # length of window in numbers of lines

content = sum([open(file).readlines() for file in glob.glob('../*/*.md') + glob.glob('../*.md')], [])
chunks = ['\n'.join(content[i: i + WINDOW]) for i in range(0, len(content), STRIDE)]
```


```python
from IPython.display import Markdown
Markdown(chunks[2])
```


```python
from superduperdb.container.document import Document

db.execute(collection.insert_many([Document({'txt': chunk}) for chunk in chunks]))
```


```python
db.execute(collection.find_one())
```


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

print(db.show('model'))
```


```python
db.show('model', 'gpt-3.5-turbo')
```


```python
from superduperdb.container.document import Document
from IPython.display import display, Markdown


q = 'Can you give me a code-snippet to set up a `VectorIndex`?'

output, context = db.predict(
    model='gpt-3.5-turbo',
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
