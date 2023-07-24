<p align="center">
   <a href="https://www.superduperdb.com">
      <picture>
         <img src=".github/logos/SuperDuperDB_logo_color.svg?raw=true" width="70%" alt="superduperdb" />
      </picture>
   </a>
</p>

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
</p>

<h3 align="center">Bring AI to your MongoDB database</h3>

<p align="center">
  <a href="https://www.mongodb.com/" target="_blank">
  <img width="250" src=".github/logos/mongodb_logo.svg">
  </a>
</p>

<p align="center">
Easily integrate AI with your data: from LLMs and public AI APIs to bespoke machine learning models and custom use-cases.
</p>
<p align="center">
<a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"><strong>Try out SuperDuperDB</strong></a>
</p>


---------------

SuperDuperDB is a Python-based open-source environment to deploy, train and operate AI models and APIs in MongoDB

---------------


```python

# Models and database clients can be converted to SuperDuperDB objects with a simple wrapper.
svm = superduper(SVM())

  

# SuperDuperDB uses MongoDB by default. SQL integrations are on the way.
db = superduper(pymongo.MongoClient().my_db)


# Once wrapped, we can fit and predict, simply specifying the data to be processed with a query.
# The model training is scheduled either on a Dask cluster or locally
coll = Collection(name='my_collection')
svm.fit(X='input_col', y='predict_col', db=db, select=coll.find({'_fold': 'train'}))

# Predictions are saved in the database alongside the inputs.
svm.predict(X='input_col', db=db, select=coll.find({'_fold': 'valid'}))

db.execute(coll.find_one())
# Document({
# "_id": ObjectId('64b6ba93f8af205501ca7748'),
# 'input_col': Encodable(x=torch.tensor([...])),
# 'output_col': {},
# '_outputs': {'input_col': {'svm': 1}}
# })

```


---------------

**What is SuperDuperDB?**

  - 🔄 an **end-to-end live AI deployment** which includes a model repository, model training and computation of outputs.
  - 📦 a **model output store** where the model outputs are stored alongside the inputs in desired formats and types.
  - 🔢 a **fully functional vector database** to easily generate vector embeddings of your data with your favorite models and APIs and connect them with MongoDB vector search.

**Who is SuperDuperDB for?**

  - **Python developers** using MongoDB who want to apply AI with simple commands.
  - **Data scientists & ML engineers** who want to develop AI models using their favourite tools, with minimum infrastructural overhead.
  -👷 **Infrastructure engineers** who want a single scalable setup that supports local, on-prem and cloud deployment.

**What can you do with SuperDuperDB?**

  - 🚀 **Deploy** all your AI models to automatically compute outputs in the database in a single environment with simple Python commands.
  - 🏋️ **Train** models on the data in your database without additional ingestion and pre-processing, simply by querying.
  - 🌐 **Integrate** APIs such as OpenAI to work together with other models on your data effortlessly.

**Why choose SuperDuperDB?**

  - 🪠 **Avoid** duplicate data, pipelines and infrastructure with a single scalable deployment.
  - 📅 Keep AI models **up-to-date** by processing new data immediately and automatically.
  - 🤸 Easy single node setup for **lightweight** use-cases.
  - 📈 **Scalable** multi-host setup for enterprise use-cases via Dask or Ray.
 
<p align="center">
  <br>
  <img width="650" src="docs/img/overview.png">
</p>

## :electric_plug: Quickstart

1. Install SuperDuperDB via `pip` (*~1 minute*): 

```
pip install superduperdb
```

2. MongoDB
    - 🔥 You already have MongoDB installed? Let's go!</li>
    - 🍃 You need to install MongoDB? See the docs [here](https://www.mongodb.com/docs/manual/installation/). (*~10-15 minutes*)
    
3. [Try out SuperDuperDB]()

### Warning

SuperDuperDB is currently in *alpha*. Please expect:

- :zap: breaking changes 
- :rock: rough edges 
- :runner: fast pace of new feature development 

## :mushroom: Developer Environment

Please see our [INSERT LINK](./) for details.

## :seedling: Contributing

Please see our [Contributing Guide](CONTRIBUTING.md) for details.
