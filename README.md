<p align="center">
	<a href="https://www.superduperdb.com">
     <img width="90%" src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/img/SuperDuperDB_logo_color.svg">
  </a>
</p>

<h1 align="center">AI with <a href="https://www.mongodb.com/">MongoDB</a>!
</h1>

<p align="center">
<a href="https://codecov.io/gh/SuperDuperDB/superduperdb/branch/main">
    <img src="https://codecov.io/gh/SuperDuperDB/superduperdb/branch/main/graph/badge.svg" alt="Coverage">
</a>
<a href="https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA">
    <img src="https://img.shields.io/badge/slack-superduperdb-8A2BE2?logo=slack" alt="slack">
</a>
<a href="https://pypi.org/project/superduperdb">
    <img src="https://img.shields.io/pypi/v/superduperdb?color=%23007ec6&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/superduperdb">
    <img src="https://img.shields.io/pypi/pyversions/superduperdb.svg" alt="Supported Python versions">
</a>    
<a href="https://colab.research.google.com/github/SuperDuperDB/superduperdb/blob/main/docs/how_to/playground.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>  
</p>


#### SuperDuperDB is an open-source environment to deploy, train and operate AI models and APIs in MongoDB using Python. <img src="https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/files/python-logo-only.svg" width="2%"/> 
#### Easily integrate AI with your data: from LLMs and public AI APIs to bespoke machine learning models and custom use-cases.
#### No data duplication, no pipelines, no duplicate infrastructure — just Python.<img src="https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/files/python-logo-only.svg" width="2%"/>

<hr>
- <a href="https://superduperdb.github.io/superduperdb/"><strong>Explore the docs!</strong></a><br>
- <a href="https://superduperdb.github.io/superduperdb/examples/index.html"><strong>Check out example use cases!</strong></a><br>
- <a href="https://colab.research.google.com/github/SuperDuperDB/superduperdb/blob/main/docs/how_to/playground.ipynb"><strong>Quickstart with Google Colab!</strong></a><br>
<hr>

<h3 align="center">
	<a href="#installation-electric_plug">Installation</a>
	<span> | </span>
	<a href="#quickstart-">Quickstart</a>
	<span> | </span>
	<a href="#contributing-seedling">Contributing </a>
	<span> | </span>
	<a href="#feedback-">Feedback </a>
	<span> | </span>
	<a href="#license-">License </a>
</h3>
<hr>


# Introduction 🔰 

### 🔮 What can you do with SuperDuperDB?

- **Deploy** all your AI models to automatically **compute outputs (inference)** in the database in a single environment with simple Python commands.  
- **Train** models on your data in your database simply by querying without additional ingestion and pre-processing.  
- **Integrate** AI APIs (such as OpenAI) to work together with other models on your data effortlessly. 
- **Search** your data with vector-search, including model management and serving.

 ### ⁉️ Why choose SuperDuperDB?

- Avoid data duplication, pipelines and duplicate infrastructure with a **single scalable deployment**.
- **Deployment always up-to-date** as new data is handled automatically and immediately.
- A **simple and familiar Python interface** that can handle even the most complex AI use-cases.

### 👨‍💻🧑‍🔬👷 Who is SuperDuperDB for?

  - **Python developers** using MongoDB who want to build AI into their applications easily.
  - **Data scientists & ML engineers** who want to develop AI models using their favourite tools, with minimum infrastructural overhead.
  - **Infrastructure engineers** who want a single scalable setup that supports both local, on-prem and cloud deployment.

### 🪄 SuperDuperDB transforms your MongoDB into:

  - **An end-to-end live AI deployment** which includes a **model repository and registry**, **model training** and **computation of outputs/ inference** 
  - **A feature store** in which the model outputs are stored alongside the inputs in any data format. 
  - **A fully functional vector database** to easily generate vector embeddings of your data with your favorite models and APIs and connect them with MongoDB or LanceDB vector search 
  - *(Coming soon)* **A model performance monitor** enabling model quality and degradation to be monitored as new data is inserted  

<br>

<p align="center">
  <img width="100%" src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/img/SuperDuperDB_diagram.svg">
</p>

# How to 🤷
### The following are examples of how you use SuperDuperDB with Python (find all how-tos and examples <a href="https://superduperdb.github.io/superduperdb">in the docs here</a>): 

- **Add a ML/AI model into your database <a href="404">(read more in the docs here)</a>:**
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

<br>

- **Train/fine-tune a model <a href="https://superduperdb.github.io/superduperdb/usage/models.html#training-models-on-data-with-fit">(read more in the docs here)</a>:**

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

<br>

- **Use MongoDB as your vector search database <a href="https://superduperdb.github.io/superduperdb/usage/vector_index.html">(read more in the docs here)</a>:**
```python
# First a "Watcher" makes sure vectors stay up-to-date
indexing_watcher = Watcher(model=OpenAIEmbedding(), key='text', select=collection.find())

# This "Watcher" is linked with a "VectorIndex"
db.add(VectorIndex('my-index', indexing_watcher=indexing_watcher))

# The "VectorIndex" may be used to search data. Items to be searched against are passed
# to the registered model and vectorized. No additional app layer is required.
# By default, SuperDuperDB uses LanceDB for vector comparison operations
db.execute(collection.like({'text': 'clothing item'}, 'my-index').find({'brand': 'Nike'}))
```

<br>

- **Use OpenAI, PyTorch or Hugging face model as an embedding model for vector search <a href="https://superduperdb.github.io/superduperdb/examples/compare_vector_search_solutions.html">(read more in the docs here)</a>:**
```python
# Create a ``VectorIndex`` instance with indexing watcher as OpenAIEmbedding and add it to the database.
db.add(
    VectorIndex(
        identifier='my-index',
        indexing_watcher=Watcher(
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

<br>

- **Add Llama 2 model directly into your database! <a href="https://superduperdb.github.io/superduperdb/usage/models.html#tranformers">(read more in the docs here)</a>:**
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

<br>

- **Use models outputs as inputs to downstream models <a href="https://superduperdb.github.io/superduperdb/usage/queries.html#featurization">(read more in the docs here)</a>:**

```python
model.predict(
    X='input_col',
    db=db,
    select=coll.find().featurize({'X': '<upstream-model-id>'}),  # already registered upstream model-id
    listen=True,
)
```

# Installation :electric_plug:

**1. Install SuperDuperDB via `pip` *(~1 minute)*:**
```
pip install superduperdb
```
#### 2. MongoDB Installation *(~10-15 minutes)*:
   - You already have MongoDB installed? Let's go!
   - You need to install MongoDB? See the docs <a href="https://www.mongodb.com/docs/manual/installation/">here</a>.

#### 3. Try one of our example use cases/notebooks <a href="https://superduperdb.github.io/superduperdb/examples/index.html">found here</a> (~as many minutes you enjoy)!
<br>

*⚠️ Disclaimer: SuperDuperDB is currently in *beta*. Please expect breaking changes, rough edges and fast pace of new and cool feature development!*


# Quickstart 🚀

#### Try SuperDuperDB in Google Colab 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuperDuperDB/superduperdb/blob/main/docs/how_to/playground.ipynb)

This will set up a playground environment that includes:
- an installation of SuperDuperDB
- an installation of a MongoDB instance containing image data and `torch` models

Have fun!

# Community & Getting Help 🙋

#### If you have any problems, questions, comments or ideas:
- Join <a href="https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA">our Slack</a> (we look forward to seeing you there).
- Search through <a href="https://github.com/SuperDuperDB/superduperdb/discussions">our GitHub Discussions</a>, or <a href="https://github.com/SuperDuperDB/superduperdb/discussions/new/choose">add a new question</a>.
- Comment <a href="https://github.com/SuperDuperDB/superduperdb/issues/">an existing issue</a> or create <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">a new one</a>.
- Feel free to contact a maintainer or community volunteer directly! 

# Contributing :seedling: 

#### There are many ways to contribute, and they are not limited to writing code. We welcome all contributions such as:

- Bug reports
- Documentation improvements
- Enhancement suggestions
- Feature requests
- Expanding the tutorials and use case examples

Please see our [Contributing Guide](CONTRIBUTING.md) for details.

# Feedback 💡

Help us to improve SuperDuperDB by providing your valuable feedback
<a href="https://docs.google.com/forms/d/e/1FAIpQLScKNyLCjSEWAwc-THjC7NJVDNRxQmVR5ey30VVayPhWOIhy1Q/viewform">here</a>!

# License 📜 

SuperDuperDB is open-source and intended to be a community effort, and it won't be possible without your support and enthusiasm.
It is distributed under the terms of the AGPL (Affero GPLv3 Public License). Any contribution made to this project will be subject to the same provisions.
