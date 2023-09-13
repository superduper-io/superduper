# Frequently asked questions

***Is SuperDuperDB a database?***

No, SuperDuperDB is an environment to run alongside your database, which makes it 🚀 **SUPER-DUPER** 🚀, by adding comprehensive AI functionality. This is embodied by our `superduper` decorator:

```python
from superduperdb import superduper

db = pymongo.MongoClient().documents

db = superduper(db)
```

***Which integrations do you have with data-bases, -lakes or -warehouses?***

We currently have first class support for MongoDB and can connect to 
the SQL databases which are supported by Ibis. This includes:

- PostgreSQL
- SQLite
- DuckDB
- Snowflake
- ClickHouse
- and many more...

***Why MongoDB and not another data store as you first data integration?***

The genesis of SuperDuperDB took place in a context where we were serving unstructured documents
at inference time. Working backwards from there, we wanted our development process to reflect
that production environment. We started with large dumps of JSON documents, but quickly 
hit a brick-wall; we deferred to hosting our own MongoDB community edition deployments 
for model development, allowing us to transition smoothly to production.

This symmetry between production and development, provides the possibility for significantly 
reduced overhead in building AI models and applications. SuperDuperDB is ***the way***
to employ such a symmetry with unstructured documents in MongoDB.

***Is SuperDuperDB an MLOps framework?***

We understand MLOps to mean DevOps for machine learning (ML) and AI.
That means focus on delivering ML and AI in conjunction with continuous integration and deployment (CI / CD), deployments defined by infrastructure as code. 

While SuperDuperDB can be used to great effect to reduce the complexity of MLOps, our starting point
is a far simpler problem setting:

:::info
Given I have AI models built as Python objects, how do I apply these to my data deployment with
zero overhead and no detours through traditional DevOps pipelines?
:::

From this point of view, SuperDuperDB is an effort to **avoid MLOps** per se. That results in 
MLOps becoming significantly simpler, the moment it becomes absolutely necessary.

***How do I deploy SuperDuperDB alongside the FARM stack?***

The [FARM stack](https://www.mongodb.com/developer/languages/python/farm-stack-fastapi-react-mongodb/)
refers to application development using FastAPI, React and MongoDB. 
In this triumvirate, MongoDB constitutes the database, the backend is deployed in Python
via FastAPI, and the frontend is built in React-Javascript. Due to the Python backend and the developments in AI in Python in 2023, SuperDuperDB is an ideal candidate to integrate here: AI models are managed by SuperDuperDB, and predictions are stored in MongoDB.

We are working on a [RESTful client-server](clientserver) implementation, allowing queries involving vector-search models to be dispatched directly from a React frontend. For applications which do not require
models at query-time, model outputs may be consumed directly via MongoDB, if [change-data-capture (CDC)](CDC) is activated. 

***Why haven't you integrated LangChain with SuperDuperDB?***

LangChain is a Python package for making:

- Data aware applications
- Allowing language models to interact with your computer system

See more [here](https://python.langchain.com/docs/get_started/introduction.html).

SuperDuperDB is focused around the database, and creating ML and AI models which operate
at the document/ row-level. This includes the possibility of seeding language models with 
context which originates from vector search. 

We found we did not need LangChain to seed OpenAI or other language models with prompts constructed out of records
from the `DB` allows us to do everything we want to do with SuperDuperDB; see [here](/docs/use_cases/items/question-the-docs) for an example of using vector-search to seed an OpenAI prompt.
This includes building highly sophisticated multimodal workflows, including interactions
between image, text, audio and more. For this reason, we decided initially not 
to integrate LangChain.
