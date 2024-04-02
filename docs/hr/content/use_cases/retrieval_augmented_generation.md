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

In order to create data, we need to create a `Schema` for encoding our special `Datatype` column(s) in the databackend.

```python
N_DATA = round(len(data) - len(data) // 4)
```


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb import Document
        
        schema = None
        
        if schema is None and datatype is None:
            data = [Document({'x': x}) for x in data]
            db.execute(table_or_collection.insert_many(data[:N_DATA]))
        elif schema is None and datatype is not None:
            data = [Document({'x': datatype(x)}) for x in data]
            db.execute(table_or_collection.insert_many(data[:N_DATA]))
        else:
            data = [Document({'x': x}) for x in data]
            db.execute(table_or_collection.insert_many(data[:N_DATA], schema='my_schema'))        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb import Document
        
        db.execute(table_or_collection.insert([Document({'x': x}) for x in data[:N_DATA]]))        
        ```
    </TabItem>
</Tabs>
```python
sample_datapoint = data[-1]
```

<!-- TABS -->
## Build text embedding model


<Tabs>
    <TabItem value="OpenAI" label="OpenAI" default>
        ```python
        %pip install openai
        
        from superduperdb.ext.openai import OpenAIEmbedding
        model = OpenAIEmbedding(identifier='text-embedding-ada-002')        
        ```
    </TabItem>
    <TabItem value="JinaAI" label="JinaAI" default>
        ```python
        %pip install jina
        
        from superduperdb.ext.jina import JinaEmbedding
         
        # define the model
        model = JinaEmbedding(identifier='jina-embeddings-v2-base-en')        
        ```
    </TabItem>
    <TabItem value="Sentence-Transformers" label="Sentence-Transformers" default>
        ```python
        from superduperdb import vector
        import sentence_transformers
        from superduperdb.ext.sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(
            identifier="embedding",
            object=sentence_transformers.SentenceTransformer("BAAI/bge-small-en"),
            datatype=vector(shape=(1024,)),
            postprocess=lambda x: x.tolist(),
            predict_kwargs={"show_progress_bar": True},
        )        
        ```
    </TabItem>
    <TabItem value="Transformers" label="Transformers" default>
        ```python
        import dataclasses as dc
        from superduperdb.components.model import _Predictor, ensure_initialized
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        @dc.dataclass(kw_only=True)
        class TransformerEmbedding(_Predictor):
            pretrained_model_name_or_path : str
        
            def init(self):
                self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
                self.model = AutoModel.from_pretrained(self.pretrained_model_name_or_path)
                self.model.eval()
        
            @ensure_initialized
            def predict_one(self, x):
                return self.predict([x])[0]
                
            @ensure_initialized
            def predict(self, dataset):
                encoded_input = self.tokenizer(dataset, padding=True, truncation=True, return_tensors='pt')
                # Compute token embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    # Perform pooling. In this case, cls pooling.
                    sentence_embeddings = model_output[0][:, 0]
                # normalize embeddings
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                return sentence_embeddings.tolist()
        
        
        model = TransformerEmbedding(identifier="embedding", pretrained_model_name_or_path="BAAI/bge-small-en")        
        ```
    </TabItem>
</Tabs>
```python
model.predict_one("What is SuperDuperDB")
```

<!-- TABS -->
## Perform a vector search

```python
from superduperdb import Document

item = Document({'x': datatype(sample_datapoint)})
```

Once we have this search target, we can execute a search as follows:


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        select = collection.find().like(sample_datapoint)        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
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
        from superduperdb.ext.openai import OpenAIChatCompletion
        
        llm = OpenAIChatCompletion(identifier='llm', model='gpt-3.5-turbo')        
        ```
    </TabItem>
    <TabItem value="Anthropic" label="Anthropic" default>
        ```python
        
        from superduperdb.ext.anthropic import AnthropicCompletions
        llm = AnthropicCompletions(identifier='llm', model='claude-2')        
        ```
    </TabItem>
    <TabItem value="vLLM" label="vLLM" default>
        ```python
        from superduperdb.ext.vllm import VllmModel
        
        predict_kwargs = {
            "max_tokens": 1024,
            "temperature": 0.8,
        }
        
        
        llm = VllmModel(
            identifier="llm",
            model_name="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
            vllm_kwargs={
                "gpu_memory_utilization": 0.7,
                "max_model_len": 10240,
                "quantization": "awq",
            },
            predict_kwargs=predict_kwargs,
        )
        
        ```
    </TabItem>
    <TabItem value="Transformers" label="Transformers" default>
        ```python
        
        from superduperdb.ext.transformers import LLM
        
        llm = LLM.from_pretrained("facebook/opt-125m", identifier="llm")        
        ```
    </TabItem>
    <TabItem value="Llama.cpp" label="Llama.cpp" default>
        ```python
        !huggingface-cli download Qwen/Qwen1.5-0.5B-Chat-GGUF qwen1_5-0_5b-chat-q8_0.gguf --local-dir . --local-dir-use-symlinks False
        
        from superduperdb.ext.llamacpp.model import LlamaCpp
        llm = LlamaCpp(identifier="llm", model_name_or_path="./qwen1_5-0_5b-chat-q8_0.gguf")        
        ```
    </TabItem>
</Tabs>
### Using LLM for text generation

```python
llm.predict_one('Tell me about the SuperDuperDB', temperature=0.7)
```

### Use in combination with Prompt

```python
from superduperdb.components.model import SequentialModel, Model

prompt_model = Model(
    identifier="prompt", object=lambda text: f"The German version of sentence '{text}' is: "
)

model = SequentialModel(identifier="The translator", predictors=[prompt_model, llm])

```

```python
model.predict_one('Tell me about SuperDuperDB')
```

