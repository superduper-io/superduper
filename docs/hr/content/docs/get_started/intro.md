---
sidebar_position: 1
---

# Introduction

## What is SuperDuperDB?

SuperDuperDB is an open-source (Apache 2.0) package, framework and environment with the mission to **bring AI to the database**.

:::tip
🔮 SuperDuperDB ***TRANSFORMS*** a database to make it a ***SUPER-DUPER*** AI tool! 🔮
:::


![](/img/SuperDuperDB_diagram.svg)

### What can you do with SuperDuperDB?

With SuperDuperDB, you can easily implement AI without the need to copy and move your data to complex MLOps pipelines and specialized vector databases. With SuperDuperDB, it's possible to integrate, train, and manage your AI models and APIs directly with your chosen database, using a simple Python interface.

- **Deploy** all your AI models to **compute outputs** on your datastore in a single environment with simple Python commands.  
- **Train** models on your data on your datastore simply by querying without additional ingestion and pre-processing.  
- **Integrate** AI APIs (such as OpenAI) to work together with other models on your data effortlessly. 
- **Search** your data with vector-search, including model management and serving.

### SuperDuperDB transforms your database into:

  - **An end-to-end live AI deployment** which includes a **model repository and registry**, **model training** and **computation of outputs/ inference** 
  - **A feature store** in which the model outputs are stored alongside the inputs in any data format. 
  - **A fully functional vector database** to easily generate vector embeddings of your data with your favorite models and APIs and connect them with your datastore (and/ or) vector database.

### What is supported?

We endeavour to provide strong support across the **database** and **AI** spectrum:

***Choose from a range of supported data-backends:***

- [**MongoDB**](https://www.mongodb.com/)
- [**PostgreSQL**](https://www.postgresql.org/)
- [**SQLite**](https://www.sqlite.org/index.html)
- [**DuckDB**](https://duckdb.org/)
- [**BigQuery**](https://cloud.google.com/bigquery)
- [**ClickHouse**](https://clickhouse.com/)
- [**DataFusion**](https://arrow.apache.org/datafusion/)
- [**Druid**](https://druid.apache.org/)
- [**Impala**](https://impala.apache.org/)
- [**MSSQL**](https://www.microsoft.com/en-us/sql-server/)
- [**MySQL**](https://www.mysql.com/)
- [**Oracle**](https://www.oracle.com/database/)
- [**pandas**](https://pandas.pydata.org/)
- [**Polars**](https://www.pola.rs/)
- [**PySpark**](https://spark.apache.org/docs/3.3.1/api/python/index.html)
- [**Snowflake**](https://www.snowflake.com/en/)
- [**Trino**](https://trino.io/)

***Choose from a range of native AI integrations:***

- [**PyTorch**](https://pytorch.org/)
- [**HuggingFace transformers**](https://huggingface.co/docs/transformers/index)
- [**Scikit-Learn**](https://scikit-learn.org/stable/)
- [**Sentence-Transformers**](https://www.sbert.net/)
- [**OpenAI**](https://openai.com/blog/openai-api)
- [**Cohere**](https://cohere.com/)
- [**Anthropic**](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)


 ### Why choose SuperDuperDB?

- Avoid data duplication, pipelines and duplicate infrastructure with a **single scalable deployment**.
- **Deployment always up-to-date** as new data is handled automatically and immediately.
- A **simple and familiar Python interface** that can handle even the most complex AI use-cases.

### Who is SuperDuperDB for?

  - **Python developers** using datastores (databases/ lakes/ warehouses) who want to build AI into their applications easily.
  - **Data scientists & ML engineers** who want to develop AI models using their favourite tools, with minimum infrastructural overhead.
  - **Infrastructure engineers** who want a single scalable setup that supports both local, on-prem and cloud deployment.

## A guide to this documentation

:::info
The connection to SuperDuperDB is referred to as `db` without further comment or explanation
throughout the documentation, with the exception of the documentation of [how to connect to SuperDuperDB](../fundamentals/04_connecting.md).
:::

Here are a few subsets of the documenation which you can follow for your specific needs.

### Do you want to get SuperDuperDB up and running?

- [Installation](installation.md)
- [Configuration](../walkthrough/01_configuration.md)
- [Connecting](../fundamentals/04_connecting.md)
- [Minimum working example](../walkthrough/05_minimum_working_example.md)

### Do you want to get started with pre-trained models quickly?

- [Predictors and models](../fundamentals/17_supported_ai_frameworks.md)
- [AI Models](../walkthrough/18_ai_models.mdx)
- [AI APIs](../walkthrough/19_ai_apis.md)
- [Applying models and predictors](../fundamentals/21_apply_models.mdx)

### Are you interested in MongoDB specific functionality?

- [Supported Query APIs](../walkthrough/11_supported_query_APIs.md)
- [Mongo Query API](../walkthrough/12_mongodb_query_API.md)
- [Change data capture](../walkthrough/32_change_data_capture.md)

### Are you interested in vector-search and document Q&A?

- [Applying models](../fundamentals/21_apply_models.mdx)
- [Vector search](../fundamentals/25_vector_search.mdx)
- [Example Q&A application](/docs/use_cases/items/question_the_docs)

### Do you want to know more about the production ready features of SuperDuperDB

- [Production mode](../walkthrough/29_developer_vs_production_mode.md)
- [CLI](../walkthrough/30_command_line_interface.md)
- [Dask integration](../walkthrough/31_non_blocking_dask_jobs.md)
- [CDC](../walkthrough/32_change_data_capture.md)
- [Vector searcher service](../walkthrough/33_vector_comparison_service.md)
