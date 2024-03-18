---
sidebar_label: Build LLM
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Build LLM


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

