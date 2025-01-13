<div align="center">
  <a href="https://www.superduper.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/superduper-io/superduper-docs/main/static/img/SuperDuperDB_logo_white.svg">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/superduper-io/superduper-docs/main/static/img/SuperDuperDB_logo_color.svg">
      <img width="50%" alt="SuperDuper logo" src="https://raw.githubusercontent.com/superduper-io/superduper-docs/main/static/img/SuperDuperDB_logo_color.svg">
    </picture>
  </a>
</div>
<div align="center">
  <h1>Build end-to-end AI-data workflows and applications with your favourite tools</h1>
</div>


<div align="center">
  <h2>
    <a href="https://docs.superduper.io"><strong>Docs</strong></a> |
    <a href="https://blog.superduper.io"><strong>Blog</strong></a> |
    <a href="https://superduper.io"><strong>Website</strong></a> |
    <a href="https://docs.superduper.io/docs/category/templates"><strong>Templates</strong></a> |
    <a href="https://join.slack.com/t/superduper-public/shared_invite/zt-1yodhtx8y-KxzECued5QBtT6JFnsSNrQ"><strong>Slack</strong></a> |
    <a href="https://www.youtube.com/channel/UC-clq9x8EGtQc6MHW0GF73g"><strong>Youtube</strong></a> |
    <a href="https://www.linkedin.com/company/superduper-io"><strong>LinkedIn</strong></a>
  </h2>
</div>

<div align="center">
  <h2>
    <a href="https://pypi.org/project/superduper-framework"><img src="https://img.shields.io/pypi/v/superduper-framework?color=%23007ec6&label=pypi%20package" alt="Package version"></a>
    <a href="https://pypi.org/project/superduper-framework"><img src="https://img.shields.io/pypi/pyversions/superduper-framework" alt="Supported Python versions"></a>    
    <a href="https://github.com/superduper-io/superduper/actions/workflows/ci_code.yml"><img src="https://github.com/superduper-io/superduper/actions/workflows/ci_code.yml/badge.svg?branch=main" /></a>
    <a href="https://github.com/superduper-io/superduper/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-green" alt="License - Apache 2.0"></a>  
  </h2>
</div>


## What is Superduper?

Superduper is a Python based framework for building **end-2-end AI-data workflows and applications** on your own data, integrating with major databases. It supports the latest technologies and techniques, including LLMs, vector-search, RAG, multimodality as well as classical AI and ML paradigms.

Developers may leverage Superduper by building **compositional and declarative objects** which out-source the details of deployment, orchestration and versioning, and more to the Superduper engine. This allows developers to completely avoid implementing MLOps, ETL pipelines, model deployment, data migration and synchronization.

Using Superduper is simply "**CAPE**": **Connect** to your data, **apply** arbitrary AI to that data, **package** and reuse the application on arbitrary data, and **execute** AI-database queries and predictions on the resulting AI outputs and data.

- **Connect**
- **Apply**
- **Package**
- **Execute**

<img src="https://github.com/superduper-io/superduper/blob/main/img/apply.gif" alt="Alt text for the image" style="width: 100%;">

**Connect**

```python
db = superduper('mongodb|postgres|mysql|sqlite|duckdb|snowflake://<your-db-uri>')
```

**Apply**

```python
listener = MyLLM('self_hosted_llm', architecture='llama-3.2', postprocess=my_postprocess).to_listener('documents', key='txt')
db.apply(listener)
```

**Package**

```python
application = Application('my-analysis-app', components=[listener, vector_index])
template = Template('my-analysis', component=app, substitutions={'documents': 'table'})
template.export('my-analysis')
```

**Execute**

```python
query = db['documents'].like({'txt', 'Tell me about Superduper'}, vector_index='my-index').select()
query.execute()
```

Superduper may be run anywhere; you can also [contact us](https://superduper.io/contact) to learn more about the enterprise platform for bringing your Superduper workflows to production at scale. 

## What does Superduper support?

Superduper is flexible enough to support a huge range of AI techniques and paradigms. We have a range of pre-built functionality in the `plugins` and `templates` directories. In particular, Superduper excels when AI and data need to interact in a continuous and tightly integrated fashion. Here are some illustrative examples, which you may try out from our templates:

- Semantic multimodal vector search ([images](https://github.com/superduper-io/superduper/tree/main/templates/multimodal_image_search), [text](https://github.com/superduper-io/superduper/tree/main/templates/text_vector_search), [video](https://github.com/superduper-io/superduper/tree/main/templates/multimodal_video_search))
- [Retrieval augmented generation](https://github.com/superduper-io/superduper/tree/main/templates/retrieval_augmented_generation) with specialized requirements (data fetching involves semantic search as well as business rules and pre-processing)
- [LLM finetuning on database hosted data](https://github.com/superduper-io/superduper/tree/main/templates/llm_finetuning)
- [Transfer learning using multimodal data](https://github.com/superduper-io/superduper/tree/main/templates/transfer_learning)

We're looking to connect with enthusiastic developers to contribute to the repertoire of amazing pre-built templates and workflows available in Superduper open-source. Please join the discussion, by contributing issues and pull requests!

## Core features

- Create a Superduper data-AI connection/ datalayer consisting of your own
  - databackend (database/ datalake/ datawarehouse)
  - metadata store (same or other as databackend)
  - artifact store (to store big objects)
  - compute implementation
- Build complex units of functionality (`Component`) using a declarative programming model, which integrate closely with data in your databackend, using a simple set of primitives and base classes.
- Build larger units of functionality wrapping several interrelated `Component` instances into an AI-data `Application`
- Reuse battle-tested `Component`, `Model` and `Application` instances using `Template`, giving developers an easy point to start with difficult AI implementations
- A transparent, human-readable, web-friendly and highly portable serialization protocol, "Superduper-protocol", to communicate results of experimentation, make `Application` lineage and versioning easy to follow, and create an elegant segway from the AI world to the databasing/ typed-data worlds.
- Execute queries using a combination of outputs of `Model` instances as well as primary databackend data, to enable the latest generation of AI-data applications, including all flavours of vector-search, RAG, and much, much more.

## Key benefits

**Massive flexibility**

Combine any Python based AI model, API from the ecosystem with the most established, battle tested databases and warehouses;  Snowflake, MongoDB, Postgres, MySQL, SQL Server, SQLite, BigQuery, and Clickhouse are all supported.

**Seamless integration avoiding MLOps**

Remove the need to implement MLOps, using the declarative and compositional Superduper components, which specify the end state that the models and data should reach.

**Promote code reusability and portability**

Package components as templates, exposing the key parameters required to reuse and communicate AI applications in your community and organization.

**Cost savings**

Implement vector search and embedding generation without requiring a dedicated vector database. Effortlessly toggle between self hosted models and API hosted models, without major code changes.

**Move to production without any additional effort**

Superduper's REST API, allows installed models to be served without additional development work. For enterprise grade scalability, fail safes, security and logging, applications and workflows created with Superduper, may be deployed in one click on [Superduper enterprise](https://superduper.io/contact).


## What's new in the `main` branch?

We are working on an upcoming release of `0.5.0`. In this release we will have:

### A graceful update schema to update already applied components

This means that changing a prompt or parameter deep in your `Component` won't mean 
starting all components from scratch. This also lays the groundwork for rollbacks 
and version pins.

### A smart form builder leveraging the `Template` class

This will allow developers to expose their applications as no-code interfaces.

### Serialization based on Python native type annotations

```python
from superduper import typing as t

class MyPDF:
    path: t.File
    my_func: t.Blob
    my_other_func: t.Pickle
```

## Getting started

**Installation**:

```bash
pip install superduper-framework
```

**View** available pre-built templates:

```bash
superduper ls
```

**Connect** and **apply** a pre-built template:

(***Note:*** *the pre-built templates are only supported by Python 3.10; you may use all of the other features in Python 3.11+.*)

```bash
# e.g. 'mongodb://localhost:27017/test_db'
SUPERDUPER_DATA_BACKEND=<your-db-uri> superduper apply simple_rag
```

**Execute** a query or prediction on the results:

```python
from superduper import superduper
db = superduper('<your-db-uri>')  # e.g. 'mongodb://localhost:27017/test_db'
db['rag'].predict('Tell me about superduper')
```

**View** and **monitor** everything in the Superduper interface. From the command line:

```bash
superduper start
```

***After doing this you are ready to build your own components, applications and templates!***

**Get started** by copying an existing template, to your own development environment:

```bash
superduper bootstrap <template_name> --destination templates/my-template
```

**Edit** the `build.ipynb` notebook, to build your own functionality.

## Currently supported datastores

- [MongoDB](https://www.mongodb.com)
- [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
- [Snowflake](https://www.snowflake.com)
- [PostgreSQL](https://www.postgresql.org)
- [MySQL](https://www.mysql.com)
- [SQLite](https://www.sqlite.org)
- [DuckDB](https://duckdb.org)
- [Google BigQuery](https://cloud.google.com/bigquery)
- [Microsoft SQL Server (MSSQL)](https://www.microsoft.com/en-us/sql-server)
- [ClickHouse](https://clickhouse.com)

## Community & getting help 

If you have any problems, questions, comments, or ideas:
- Join <a href="https://join.slack.com/t/superduper-public/shared_invite/zt-1yodhtx8y-KxzECued5QBtT6JFnsSNrQ">our Slack</a> (we look forward to seeing you there).
- Search through <a href="https://github.com/superduper-io/superduper/discussions">our GitHub Discussions</a>, or <a href="https://github.com/superduper-io/superduper/discussions/new/choose">add a new question</a>.
- Comment <a href="https://github.com/superduper-io/superduper/issues/">an existing issue</a> or create <a href="https://github.com/superduper-io/superduper/issues/new/choose">a new one</a>.
- Help us to improve Superduper by providing your valuable feedback <a href="https://github.com/superduper-io/superduper/discussions/new/choose">here</a>!
- Email us at `gethelp@superduper.io`.
- Visit our [YouTube channel](https://www.youtube.com/@superduper-io).
- Follow us on [Twitter (now X)](https://twitter.com/superduperdb).
- Connect with us on [LinkedIn](https://www.linkedin.com/company/superduper-io).
- Feel free to contact a maintainer or community volunteer directly! 


## Contributing  
There are many ways to contribute, and they are not limited to writing code. We welcome all contributions such as:

- <a href="https://github.com/superduper-io/superduper/issues/new/choose">Bug reports</a>
- <a href="https://github.com/superduper-io/superduper/issues/new/choose">Documentation improvements</a>
- <a href="https://github.com/superduper-io/superduper/issues/new/choose">Enhancement suggestions</a>
- <a href="https://github.com/superduper-io/superduper/issues/new/choose">Feature requests</a>
- <a href="https://github.com/superduper-io/superduper/issues/new/choose">Expanding the tutorials and use case examples</a>

Please see our [Contributing Guide](CONTRIBUTING.md) for details.


## Contributors
Thanks goes to these wonderful people:

<a href="https://github.com/superduper-io/superduper/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=superduperdb/superduper" />
</a>


## License  

Superduper is open-source and intended to be a community effort, and it wouldn't be possible without your support and enthusiasm.
It is distributed under the terms of the Apache 2.0 license. Any contribution made to this project will be subject to the same provisions.

## Join Us 
We are looking for nice people who are invested in the problem we are trying to solve to join us full-time. Find roles that we are trying to fill <a href="https://join.com/companies/superduper">here</a>!
