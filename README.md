<p align="center">
  <a href="https://www.superduperdb.com">
    <img width="90%" src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/static/img/SuperDuperDB_logo_color.svg">
  </a>
</p>

<div align="center">
	
# Bring AI to your favourite database! 
 

</div>


<div align="center">
	
### <a href="https://superduperdb.github.io/superduperdb/"><strong>Docs</strong></a> | <a href="https://docs.superduperdb.com/blog"><strong>Blog</strong></a> | <a href="https://docs.superduperdb.com/docs/category/use-cases"><strong>Example Use-Cases & Apps</strong></a> | <a href="http://demo.superduperdb.com"><strong>Jupyter Demo</strong></a>

</div>

 
<p align="center">
	<a href="https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA">
    <img src="https://img.shields.io/badge/slack-superduperdb-8A2BE2?logo=slack" alt="slack">
</a>
<a href="https://codecov.io/gh/SuperDuperDB/superduperdb/branch/main">
    <img src="https://codecov.io/gh/SuperDuperDB/superduperdb/branch/main/graph/badge.svg" alt="Coverage">
</a>
<a href="https://pypi.org/project/superduperdb">
    <img src="https://img.shields.io/pypi/v/superduperdb?color=%23007ec6&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/superduperdb">
    <img src="https://img.shields.io/pypi/pyversions/superduperdb.svg" alt="Supported Python versions">
</a>    
<a href="https://en.wikipedia.org/wiki/Apache_License#Apache_License_2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg", alt="Apache License">
</a> 
</p>

<div align="center">	
	
 `🔮 SuperDuperDB is open-source: Leave a star ⭐️ to support the project!`
 </div>

# Overview

SuperDuperDB is not just another database; it's a revolutionary platform that transforms your existing database into an AI powerhouse. 

With SuperDuperDB, you can enjoy:

- **A single scalable AI deployment** of all your models and AI APIs including output computation (inference) — always up-to-date as changing data is handled automatically and immediately.
- **A model trainer** allowing you to easily train and fine-tune models simply by querying your database.
- **A feature store** in which the model outputs are stored alongside the inputs in any data format.
- **A fully functional vector database** to easily generate vector embeddings of your data with your favorite models and APIs and connect them with your database (and/ or) vector database.


<p align="center">
  <img width="90%" src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/static/img/SuperDuperDB_diagram.svg">
</p>



### Current Integrations (*more coming soon*):

Build next-gen AI applications using Python without the need for complex MLOps pipelines, infrastructure, or data duplication and migration to specialized vector databases.

<div align="center">	


| Databases                 |             | AI Frameworks            |             | Models & AI APIs |
|---------------------------|-------------|--------------------------|-------------|------------------|
| MongoDB                   |             | PyTorch                  |             | OpenAI           |
| MongoDB Atlas             |             | Scikit-Learn             |             | Cohere           |
| AWS S3                    |             | HuggingFace Transformers |             | Anthropic        |
| PostgreSQL (experimental) |             |                          |             |                  |
| SQLite (experimental)     |             |                          |             |                  |
| DuckDB (experimental)     |             |                          |             |                  |
| MySQL (experimental)      |             |                          |             |                  |
| Snowflake (experimental)  |             |                          |             |                  |

</div>

### Featured Examples

Accelerate AI development and enable data symbiotic AI applications with a simple and familiar Python<img src="https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/files/python-logo-only.svg" width="2%"/> interface that can support even the most complex AI use-cases.

- from LLM-based (RAG) chatbots and vector search
- image generation, segmentation, time series forecasting, anomaly detection, classification, recommendation, personalisation etc.
- to highly custom machine learning use-cases and workflows



<table>
  <tr>
    <td width="30%">
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/examples/multimodal_image_search_clip.ipynb">
        <img src="./apidocs/img/image-search.png" />
      </a>
    </td>
    <td width="30%">
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/examples/video_search.ipynb">
        <img src="./apidocs/img/video-search.png" />
      </a>
    </td>
    <td width="30%">
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/examples/openai.ipynb">
        <img src="./apidocs/img/document-search.png" />
      </a>
    </td>
  </tr>
  <tr>
    <th>
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/examples/multimodal_image_search_clip.ipynb">Text-To-Image search </a>
    </th>
    <th>
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/examples/video_search.ipynb">Video search</a>
    </th>
    <th>
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/examples/openai.ipynb">Document search</a>
    </th>
  </tr>
  <tr>     
    <td width="30%">
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/examples/voice_memos.ipynb">
        <img src="./apidocs/img/voice-search.png" />
      </a>
    </td>
    <td width="30%">
      <a href="URL_TO_FIFTH_CATEGORY">
        <img src="IMAGE_URL_FOR_FIFTH_CATEGORY" />
      </a>
    </td>
    <td width="30%">
      <a href="URL_TO_SIXTH_CATEGORY">
        <img src="IMAGE_URL_FOR_SIXTH_CATEGORY" />
      </a>
    </td>
  </tr>
  <tr>
    <th>
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/examples/voice_memos.ipynb">Question your memos</a>
    </th>
    <th>
      <a href="URL_TO_FIFTH_CATEGORY">Fifth Category</a>
    </th>
    <th>
      <a href="URL_TO_SIXTH_CATEGORY">Sixth Category</a>
    </th>
  </tr>
</table>



# Quickstart


## Installation
#### 1. Install SuperDuperDB via `pip` *(~1 minute)*
```
pip install superduperdb
```
#### 2. Database installation (for MongoDB) *(~10-15 minutes)*:
   - You already have MongoDB installed? Let's go!
   - You need to install MongoDB? See the docs <a href="https://www.mongodb.com/docs/manual/installation/">here</a>.

#### 3. Now you can start using SuperDuperDB with oneclick: [Live Demo](https://demo.superduperdb.com)!



## Tutorial

In this tutorial, you will learn how to Integrate, train and manage any AI models and APIs directly with your database with your data. You can visit the <a href="https://docs.superduperdb.com/docs/docs/intro">docs</a> to learn more.


#### - Deploy ML/AI models to your database:
Automatically compute outputs (inference) with your database in a single environment.

```python
import pymongo
from sklearn.svm import SVC

from superduperdb import superduper

# Make your db superduper!
db = superduper(pymongo.MongoClient().my_db)

# Models client can be converted to SuperDuperDB objects with a simple wrapper.
model = superduper(SVC())

# Add the model into the database
db.add(model)

# Predict on the selected data.
model.predict(X='input_col', db=db, select=Collection(name='test_documents').find({'_fold': 'valid'}))
```


#### - Train models directly from your database.
Simply query your database, without additional ingestion and pre-processing:

```python
import pymongo
from sklearn.svm import SVC

from superduperdb import superduper

# Make your db superduper!
db = superduper(pymongo.MongoClient().my_db)

# Models client can be converted to SuperDuperDB objects with a simple wrapper.
model = superduper(SVC())

# Predict on the selected data.
model.predict(X='input_col', db=db, select=Collection(name='test_documents').find({'_fold': 'valid'}))
```

#### - Vector-Search your data:
Use your existing favorite database as a vector search database, including model management and serving. 

```python
# First a "Listener" makes sure vectors stay up-to-date
indexing_listener = Listener(model=OpenAIEmbedding(), key='text', select=collection.find())

# This "Listener" is linked with a "VectorIndex"
db.add(VectorIndex('my-index', indexing_listener=indexing_listener))

# The "VectorIndex" may be used to search data. Items to be searched against are passed
# to the registered model and vectorized. No additional app layer is required.
# By default, SuperDuperDB uses LanceDB for vector comparison operations
db.execute(collection.like({'text': 'clothing item'}, 'my-index').find({'brand': 'Nike'}))
```

#### - Integrate AI APIs to work together with other models. 
Use OpenAI, PyTorch or Hugging face model as an embedding model for vector search.

```python
# Create a ``VectorIndex`` instance with indexing listener as OpenAIEmbedding and add it to the database.
db.add(
    VectorIndex(
        identifier='my-index',
        indexing_listener=Listener(
            model=OpenAIEmbedding(identifier='text-embedding-ada-002'),
            key='abstract',
            select=Collection(name='wikipedia').find(),
        ),
    )
)
# The above also executes the embedding model (openai) with the select query on the key.

# Now we can use the vector-index to search via meaning through the wikipedia abstracts
cur = db.execute(
    Collection(name='wikipedia')
        .like({'abstract': 'philosophers'}, n=10, vector_index='my-index')
)
```


#### - Add a Llama 2 model directly into your database!:
```python
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = Pipeline(
    identifier='my-sentiment-analysis',
    task='text-generation',
    preprocess=tokenizer,
    object=pipeline,
    torch_dtype=torch.float16,
    device_map="auto",
)

# You can easily predict on your collection documents.
model.predict(
    X=Collection(name='test_documents').find(),
    db=db,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200
)
```

#### - Use models outputs as inputs to downstream models:

```python
model.predict(
    X='input_col',
    db=db,
    select=coll.find().featurize({'X': '<upstream-model-id>'}),  # already registered upstream model-id
    listen=True,
)
```



# Community & Getting Help 

#### If you have any problems, questions, comments or ideas:
- Join <a href="https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA">our Slack</a> (we look forward to seeing you there).
- Search through <a href="https://github.com/SuperDuperDB/superduperdb/discussions">our GitHub Discussions</a>, or <a href="https://github.com/SuperDuperDB/superduperdb/discussions/new/choose">add a new question</a>.
- Comment <a href="https://github.com/SuperDuperDB/superduperdb/issues/">an existing issue</a> or create <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">a new one</a>.
- Send us an email to gethelp@superduperdb.com.
- Feel free to contact a maintainer or community volunteer directly! 

# Contributing  

#### There are many ways to contribute, and they are not limited to writing code. We welcome all contributions such as:


- <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">Bug reports</a>
- <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">Documentation improvements</a>
- <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">Enhancement suggestions</a>
- <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">Feature requests</a>
- <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">Expanding the tutorials and use case examples</a>

Please see our [Contributing Guide](CONTRIBUTING.md) for details.

# Feedback 

Help us to improve SuperDuperDB by providing your valuable feedback
<a href="https://docs.google.com/forms/d/e/1FAIpQLScKNyLCjSEWAwc-THjC7NJVDNRxQmVR5ey30VVayPhWOIhy1Q/viewform">here</a>!

# License  

SuperDuperDB is open-source and intended to be a community effort, and it won't be possible without your support and enthusiasm.
It is distributed under the terms of the Apache 2.0 license. Any contribution made to this project will be subject to the same provisions.

# Join Us 

We are looking for nice people who are invested in the problem we are trying to solve to join us full-time. Find roles that we are trying to fill <a href="https://join.com/companies/superduperdb">here</a>!
