---
sidebar_label: Build text embedding model
filename: build_text_embedding_model.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Build text embedding model


<Tabs>
    <TabItem value="OpenAI" label="OpenAI" default>
        ```python
        !pip install openai
        from superduperdb.ext.openai import OpenAIEmbedding
        
        embedding_model = OpenAIEmbedding(identifier='text-embedding-ada-002')        
        ```
    </TabItem>
    <TabItem value="JinaAI" label="JinaAI" default>
        ```python
        import os
        from superduperdb.ext.jina import JinaEmbedding
        
        os.environ["JINA_API_KEY"] = "jina_xxxx"
         
        # define the model
        embedding_model = JinaEmbedding(identifier='jina-embeddings-v2-base-en')        
        ```
    </TabItem>
    <TabItem value="Sentence-Transformers" label="Sentence-Transformers" default>
        ```python
        !pip install sentence-transformers
        from superduperdb import vector
        import sentence_transformers
        from superduperdb.ext.sentence_transformers import SentenceTransformer
        
        embedding_model = SentenceTransformer(
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
        from superduperdb import vector
        from superduperdb.components.model import Model, ensure_initialized, Signature
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        class TransformerEmbedding(Model):
            signature: Signature = 'singleton'
            pretrained_model_name_or_path : str
        
            def init(self):
                self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
                self.model = AutoModel.from_pretrained(self.pretrained_model_name_or_path)
                self.model.eval()
        
            @ensure_initialized
            def predict(self, x):
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
        
        
        embedding_model = TransformerEmbedding(identifier="embedding", pretrained_model_name_or_path="BAAI/bge-small-en", datatype=vector(shape=(384, )))        
        ```
    </TabItem>
</Tabs>
```python
print(len(embedding_model.predict("What is SuperDuperDB")))
```

<DownloadButton filename="build_text_embedding_model.md" />
