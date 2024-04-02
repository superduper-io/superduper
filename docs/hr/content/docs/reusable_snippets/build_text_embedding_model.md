---
sidebar_label: Build text embedding model
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Build text embedding model


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
        %pip install sentence-transformers
        
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
        %pip install transformers torch
        
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

