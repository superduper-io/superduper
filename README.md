<p align="center">
<a href="https://github.com/SuperDuperDB/superduperdb-stealth/actions?query=workflow%3Aci+event%3Apush+branch%3Amain" target="_blank">
    <img src="https://github.com/SuperDuperDB/superduperdb-stealth/workflows/CI/badge.svg?event=push&branch=main" alt="CI">
</a>
<a href="https://codecov.io/gh/SuperDuperDB/superduperdb-stealth/branch/main" target="_blank">
    <img src="https://codecov.io/gh/SuperDuperDB/superduperdb-stealth/branch/main/graph/badge.svg" alt="Coverage">
</a>
<a href="https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA" target="_blank">
    <img src="https://img.shields.io/badge/slack-superduperdb-8A2BE2?logo=slack" alt="slack">
</a>
<a href="https://pypi.org/project/superduperdb" target="_blank">
    <img src="https://img.shields.io/pypi/v/superduperdb?color=%23007ec6&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/superduperdb" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/superduperdb.svg" alt="Supported Python versions">
</a>    
<a href="https://colab.research.google.com/github/SuperDuperDB/superduperdb-stealth/blob/main/notebooks/playground.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>  

</p>

<p align="center">
   <a href="https://www.superduperdb.com">
         <img src="docs/img/1680x420_Header_Logo.png?raw=true" width="100%" alt="SuperDuperDB" />
   </a>

</p>

<h1 align="center">Bring AI to your <a href="https://www.mongodb.com/" target="_blank">MongoDB</a>-based application!
</h1>

#### SuperDuperDB is an open-source environment to deploy, train and operate AI models and APIs in MongoDB using Python <img src="https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/files/python-logo-only.svg" width="2%"/>. 
#### Easily integrate AI with your data: from LLMs and public AI APIs to bespoke machine learning models and custom use-cases.

#### No data duplication, no pipelines, no duplicate infrastructure — just Python.<img src="https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/files/python-logo-only.svg" width="2%"/>

<hr>
- <a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"><strong>Exlore the docs!</strong></a><br>
- <a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"><strong>Check out example use cases!</strong></a><br>
- <a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"><strong>Quickstart with Google Colab!</strong></a><br>

<hr>

# Introduction 🔰 


### 🔮 What can you do with SuperDuperDB?

- **Deploy** all your AI models to automatically **compute outputs (inference)** in the database in a single environment with simple Python commands.  
- **Train** models on your data in your database simply by querying without additional ingestion and pre-processing.  
- **Integrate** AI APIs (such as OpenAI) to work together with other models on your data effortlessly. 

 ### ⁉️ Why choose SuperDuperDB?

- Avoid data duplication, pipelines and duplicate infrastructure with a single **scalable** deployment.
- **Deployment always up-to-date** as new data is handled automatically and immediately.
- **Python only**: Empowering developers to implement robust AI use-cases, standing the test of time.

### 👨‍💻🧑‍🔬👷 Who is SuperDuperDB for?

  - **Python developers** using MongoDB who want to build AI into their applications easily.
  - **Data scientists & ML engineers** who want to develop AI models using their favourite tools, with minimum infrastructural overhead.
  - **Infrastructure engineers** who want a single scalable setup that supports both local, on-prem and cloud deployment.

### 🪄 SuperDuperDB transforms our MongoDB into:

  - **An end-to-end live AI deployment** which includes a **model repository and registry**, **model training** and **computation of outputs/ inference** 
  - **A feature store** where the model outputs are stored alongside the inputs in desired formats and types 
  - **A fully functional vector database** to easily generate vector embeddings of your data with your favorite models and APIs and connect them with MongoDB vector search 
  - *(Coming soon)* **A model performance monitor** enabling model quality and degradation to be monitored as new data is inserted  


<p align="center">
  <br>
  <img width="650" src="docs/img/overview.png">
</p>

# How to 🤷
#### The following are three examples of how you use SuperDuperDB in Python (find all how-tos <a href="404" target="_blank">in the docs here</a>): 

- **Add a ML/DL model into your database <a href="404" target="_blank">(read more in the docs here)</a>:**
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
- **Train/Fine-tune a model <a href="404" target="_blank">(read more in the docs here)</a>:**
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

- **Use MongoDB as your vector search database <a href="404" target="_blank">(read more in the docs here)</a>:**
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

- **Use OpenAI as embedding model for vector search <a href="404" target="_blank">(read more in the docs here)</a>:**
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

- **Add Llama 2 model directly into your database! <a href="404" target="_blank">(read more in the docs here)</a>:**
```python
model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
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

- **Create downstream classifier model <a href="404" target="_blank">(read more in the docs here)</a>:**
```python
model.fit(
    X='text',
    y='label',
    db=db,
    select=collection.find().featurize({'text': '<my-upstream-model>'}),
)
```

# Installation :electric_plug:

**1. Install SuperDuperDB via `pip` *(~1 minute)*:**
```
pip install superduperdb
```
#### 2. MongoDB Installation *(~10-15 minute)*:
   - You already have MongoDB installed? Let's go!
   - You need to install MongoDB? See the docs <a href="https://www.mongodb.com/docs/manual/installation/">here</a>.

#### 3. Try one of our example use cases/ notebooks <a href="404">found here!</a> (~as many minutes you enjoy)
<br>

*⚠️ Disclaimer: SuperDuperDB is currently in *alpha*. Please expect breaking changes, rough edges and fast pace of new feature development*


# Quickstart 🚀

Try SuperDuperDB in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11SJunSZc52jUuYrmNi5GziombcQ_sdhJ#scrollTo=XwWu32JBovja)

This will set up a playground demo environment:
- an installation of SuperDuperDB
- an installation of a MongoDB instance containing Youtube transcripts

Enjoy and have fun with it! 🎊


# Community & Getting Help 🙋

If you have any problems, questions, commets or ideas:
- Join <a href="https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA" target="_blank">our Slack</a> (we look forward to seeing you there 💜).
- Search through <a href="404" target="_blank">our GitHub Discussions</a>, or <a href="404" target="_blank">add a new question</a>.
- Comment <a href="404" target="_blank">an existing issue</a> or create <a href="404" target="_blank">
a new one</a>.
- Feel free to contact a maintainer or community volunteer directly! 


# Contributing :seedling: 

There are many ways to contribute, and they are not limited to writing code. We welcome all contributions such as:
- Bug reports
- Documentation improvements
- Enhancement suggestions
- Feature requests
- Expanding the tutorials and use case examples

Please see our [Contributing Guide](CONTRIBUTING.md) for details.


# License 📜 

SuperDuperDB is open-source and intended to be a community effort, and it won't be possible without your support and enthusiasm.
It is distributed under the terms of the Apache License Version 2.0. Any contribution made to this project will be licensed under the Apache License Version 2.0.
