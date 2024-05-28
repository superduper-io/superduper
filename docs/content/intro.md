# Welcome to SuperDuperDB!

Hi 👋 and welcome to the open-source SuperDuperDB project! If you 
are reading this, you are probably interested in taking full control 
of your AI-data integrations, and would like to leverage the full power
of the open-source AI ecosystem on your databases. Well done! We agree
that herein lies the path to avoiding vendor lock-in, ensuring 
proper compliance with data and AI regulation, as well as unlocking 
brand new functionality which you will build by combining best-in-class
open-source AI projects.

### What is SuperDuperDB?

:::tip
SuperDuperDB is **a virtual AI-datalayer** where AI-models can interoperate
directly and scalably on data in databases, datalakes, datawarehouses, 
with zero-additional overhead compared to local computation on in-memory data.

**SuperDuperDB is open-sourced in Python under the Apache 2.0 license**
:::

With SuperDuperDB, developers may:

  - connect an AI development environment directly to data
  - connect an AI production environment directly to data
  - create their own flexible platform connecting your AI and data for all AI stakeholders to collaborate on

### SuperDuperDB can handle classical AI/ machine learning paradigms...

- classification
- regression
- forecasting
- clustering
- *and much, more more...*

### As well as the most update to date techniques...

- generative AI
- LLMs
- retrieval augmented generation (RAG)
- computer vision
- multimodal AI
- *and much, more more...*

![](/img/superduperdb.gif)

### What problem does SuperDuperDB solve?

AI development consists of multiple phases, tooling universes, stakeholders:

***Phases***

- Data injestion & preparation
- Model development and training
- Production computation, inference and fine-tuning

***Tooling***

- Database, lake, warehouse, object storage
- IDEs, notebooks, software packages
- ETL jobs, cloud compute

***Stakeholders***

- AI researchers
- Data scientists and analysts
- Engineers: MLOps, cloud
- Decision makers

:::important
    A central problem in operationalizing AI is that the phases, tooling and stakeholders
    do not have a ***single accepted environment to co-exist, collaborate and interface*** which fits 
    developers' and organizations' operational needs.
:::

For more information about SuperDuperDB and why we believe it is much needed, [read this blog post](https://blog.superduperdb.com/superduperdb-the-open-source-framework-for-bringing-ai-to-your-datastore/). 

### How can developers use SuperDuperDB?

SuperDuperDB boils down to 3 key patterns:

#### 1. Connect to your data

```python
from superduperdb import superduper

db = superduper('<your-database-uri>')
```

#### 2. Apply AI to your data

```python

component = ...   # build your AI with anything from the 
                  # python ecosystem

db.apply(component)
```

#### 3. Query your data to obtain predictions, select data or perform vector-searches

```python
db.execute(query)
```

### What does apply AI to data mean?

"Applying AI" to data can mean numerous things, which developers 
are able to determine themselves. Any of these things is possible:

- Compute outputs on incoming data
- Train a model on database data
- Configure vector-search on database
- Measure the performance of models
- Configure models to work together

### Why is the "DB" so important in AI?

SuperDuperDB uses the fact that AI development always starts with data, ends with data, and interfaces 
with data from conception, to productionized deployment. Any environment which has a chance of uniting 
the diverse tools and stakeholders involved in AI development, needs a single way 
for AI models and algorithms to be connected to data. ***That way is SuperDuperDB***.

:::important
By integrating AI directly at data's source, SuperDuperDB enables developers to avoid implementing MLops.
:::

### What integrations does SuperDuperDB include?

#### Data

- MongoDB
- PostgreSQL
- SQLite
- Snowflake
- MySQL
- Oracle
- MSSQL
- Clickhouse
- Pandas

#### AI frameworks

- OpenAI
- Cohere
- Anthropic
- PyTorch
- Sklearn
- Transformers
- Sentence-Transformers

### What important additional aspects does SuperDuperDB include?

Developers may:

- Choose whether to deploy SuperDuperDB in single blocking process or in scalable, non-blocking mode via `ray`
- Choose whether to use their own self-programmed home grown models, or integrate AI APIs and open-source frameworks
- Choose which type of data they use, including images, videos, audio, or custom datatypes
- Automatically version and track all functionality they use
- Keep control over which data is exposed to API services (if any) by leveraging model self-hosting

### Key Features:

- **[Integration of AI with your existing data infrastructure](https://docs.superduperdb.com/docs/docs/walkthrough/apply_models):** Integrate any AI models and APIs with your databases in a single scalable deployment without the need for additional pre-processing steps, ETL, or boilerplate code.
- **[Streaming Inference](https://docs.superduperdb.com/docs/docs/walkthrough/daemonizing_models_with_listeners):** Have your models compute outputs automatically and immediately as new data arrives, keeping your deployment always up-to-date.
- **[Scalable Model Training](https://docs.superduperdb.com/docs/docs/walkthrough/training_models):** Train AI models on large, diverse datasets simply by querying your training data. Ensured optimal performance via in-build computational optimizations.
- **[Model Chaining](https://docs.superduperdb.com/docs/docs/walkthrough/linking_interdependent_models/)**: Easily set up complex workflows by connecting models and APIs to work together in an interdependent and sequential manner.
- **[Simple, but Extendable Interface](https://docs.superduperdb.com/docs/docs/fundamentals/procedural_vs_declarative_api)**: Add and leverage any function, program, script, or algorithm from the Python ecosystem to enhance your workflows and applications. Drill down to any layer of implementation, including the inner workings of your models, while operating SuperDuperDB with simple Python commands.
- **[Difficult Data Types](https://docs.superduperdb.com/docs/docs/walkthrough/encoding_special_data_types/)**: Work directly in your database with images, video, audio, and any type that can be encoded as `bytes` in Python.
- **[Feature Storing](https://docs.superduperdb.com/docs/docs/walkthrough/encoding_special_data_types):** Turn your database into a centralized repository for storing and managing inputs and outputs of AI models of arbitrary data types, making them available in a structured format and known environment.
- **[Vector Search](https://docs.superduperdb.com/docs/docs/walkthrough/vector_search):** No need to duplicate and migrate your data to additional specialized vector databases - turn your existing battle-tested database into a fully-fledged multi-modal vector-search database, including easy generation of vector embeddings and vector indexes of your data with preferred models and APIs.