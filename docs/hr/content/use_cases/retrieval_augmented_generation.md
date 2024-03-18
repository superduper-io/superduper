---
sidebar_label: Retrieval augmented generation
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Retrieval augmented generation

The first step in any SuperDuperDB application is to connect to your data-backend with SuperDuperDB:

<!-- TABS -->
## Connect to SuperDuperDB


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb import superduper
        
        db = superduper('mongodb://localhost:27017/documents')        
        ```
    </TabItem>
    <TabItem value="SQLite" label="SQLite" default>
        ```python
        from superduperdb import superduper
        
        db = superduper('sqlite://my_db.db')        
        ```
    </TabItem>
</Tabs>
Once you have done that you are ready to define your datatype(s) which you would like to "search".

<!-- TABS -->
## Insert data

In order to create data, we need create a `Schema` for encoding our special `Datatype` column(s) in the databackend.

Here's some sample data to work with:


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        !curl -O https://jupyter-sessions.s3.us-east-2.amazonaws.com/text.json
        
        import json
        with open('text.json') as f:
            data = json.load(f)        
        ```
    </TabItem>
    <TabItem value="Images" label="Images" default>
        ```python
        !curl -O https://jupyter-sessions.s3.us-east-2.amazonaws.com/images.zip
        !unzip images.zip
        
        import os
        data = [{'image': f'file://image/{file}'} for file in os.listdir('./images')]        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        !curl -O https://jupyter-sessions.s3.us-east-2.amazonaws.com/audio.zip
        !unzip audio.zip
        
        import os
        data = [{'audio': f'file://audio/{file}'} for file in os.listdir('./audio')]        
        ```
    </TabItem>
</Tabs>
The next code-block is only necessary if you're working with a custom `DataType`:

```python
from superduperdb import Schema, Document

schema = Schema(
    'my_schema',
    fields={
        'my_key': dt
    }
)

data = [
    Document({'my_key': item}) for item in data
]
```


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb.backends.mongodb import Collection
        
        collection = Collection('documents')
        
        db.execute(collection.insert_many(data))        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb.backends.ibis import Table
        
        table = Table(
            'my_table',
            schema=schema,
        )
        
        db.add(table)
        db.execute(table.insert(data))        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Build text embedding model


<Tabs>
    <TabItem value="OpenAI" label="OpenAI" default>
        ```python
        ...        
        ```
    </TabItem>
    <TabItem value="JinaAI" label="JinaAI" default>
        ```python
        ...        
        ```
    </TabItem>
    <TabItem value="Sentence-Transformers" label="Sentence-Transformers" default>
        ```python
        ...        
        ```
    </TabItem>
    <TabItem value="Transformers" label="Transformers" default>
        ```python
        ...        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Perform a vector search

- `item` is the item which is to be encoded
- `dt` is the `DataType` instance to apply

```python
from superduperdb import Document

item = Document({'my_key': dt(item)})
```

Once we have this search target, we can execute a search as follows:


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb.backends.mongodb import Collection
        
        collection = Collection('documents')
        
        select = collection.find().like(item)        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        
        # Table was created earlier, before preparing vector-search
        table = db.load('table', 'documents')
        
        select = table.like(item)        
        ```
    </TabItem>
</Tabs>
```python
results = db.execute(select)
```

<!-- TABS -->
## Build LLM


<Tabs>
    <TabItem value="OpenAI" label="OpenAI" default>
        ```python
        
        ...        
        ```
    </TabItem>
    <TabItem value="Anthropic" label="Anthropic" default>
        ```python
        
        ...        
        ```
    </TabItem>
    <TabItem value="vLLM" label="vLLM" default>
        ```python
        
        ...        
        ```
    </TabItem>
    <TabItem value="Transformers" label="Transformers" default>
        ```python
        
        ...        
        ```
    </TabItem>
    <TabItem value="Llama.cpp" label="Llama.cpp" default>
        ```python
        
        ...        
        ```
    </TabItem>
</Tabs>
```python
llm.predict_one(X='Tell me about SuperDuperDB')
```


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb.components.model import QueryModel
        from superduperdb import Variable, Document
        
        query_model = QueryModel(
            select=collection.find().like(Document({'my_key': Variable('item')}))
        )        
        ```
    </TabItem>
</Tabs>
```python
from superduperdb.components.graph import Graph, Input
from superduperdb import superduper


@superduper
class PromptBuilder:
    def __init__(self, initial_prompt, post_prompt, key):
        self.inital_prompt = initial_prompt
        self.post_prompt = post_prompt
        self.key = key

    def __call__(self, X, context):
        return (
            self.initial_prompt + '\n\n'
            + [r[self.key] for r in context]
            + self.post_prompt + '\n\n'
            + X
        )


prompt_builder = PromptBuilder(
    initial_prompt='Answer the following question based on the following facts:',
    post_prompt='Here\'s the question:',
    key='my_key',
)

with Graph() as G:
    input = Input('X')
    query_results = query_model(item=input)
    prompt = prompt_builder(X=input, context=query_results)
    output = llm(X=prompt)
```

