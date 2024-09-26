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
  <h1>Bring AI to your favourite database</h1>
</div>

<div align="center">
  <h2>
    <a href="https://docs.superduper.io"><strong>Docs</strong></a> |
    <a href="https://blog.superduper.io"><strong>Blog</strong></a> |
    <a href="https://superduper.io"><strong>Website</strong></a> |
    <a href="https://docs.superduper.io/docs/category/use-cases"><strong>Use-Cases</strong></a> |
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

Superduper (formerly SuperDuperDB) is a Python framework for integrating AI models and workflows with major databases. Implement custom AI solutions without moving your data through complex pipelines and specialized vector databases, including hosting of your own models, streaming inference and scalable model training/fine-tuning.

Transform your existing database into an AI development and deployment stack with one command, streamlining your AI workflows in one environment instead of being spread across systems and environments:
```
db = superduper('mongodb|postgres|mysql|sqlite|duckdb|snowflake://<your-db-uri>')
```

Run Superduper anywhere, or [contact us](https://superduper.io/contact) to learn more about the enterprise platform for bringing your apps to production at scale. 



## Key features
- **[Integration of AI with your existing data infrastructure](https://docs.superduper.io/docs/apply_api/model):** Integrate any AI models and APIs with your databases in a single environment, without the need for additional pre-processing steps, ETL or boilerplate code.
- **[Inference via change-data-capture](https://docs.superduper.io/docs/models/daemonizing_models_with_listeners):** Have your models compute outputs automatically and immediately as new data arrives, keeping your deployment always up-to-date.
- **[Scalable model hosting](https://docs.superduper.io/docs/category/ai-integrations):** Host your own models from form HuggingFace, PyTorch and scikit-learn and safeguard your data.
- **[Scalable model training](https://docs.superduper.io/docs/models/training_models):** Train AI models on large, diverse datasets simply by querying your training data. Ensured optimal performance via in-build computational optimizations.
- **[Model chaining](https://docs.superduper.io/docs/models/linking_interdependent_models)**: Easily setup complex workflows by connecting models and APIs to work together in an interdependent and sequential manner.
- **[Simple Python interface](https://docs.superduper.io/docs/core_api/intro)**: Replace writing thousand of lines of glue code with simple Python commands, while being able to drill down to any layer of implementation detail, like the inner workings of your models or your training details.
- **[Python-first](https://docs.superduper.io/docs/fundamentals/class_hierarchy)**: Bring and leverage any function, program, script or algorithm from the Python ecosystem to enhance your workflows and applications.
- **[Difficult data-types](https://docs.superduper.io/docs/reusable_snippets/create_datatype)**: Work directly with images, video, audio in your database, and any type which can be encoded as `bytes` in Python.
- **[Feature storing](https://docs.superduper.io/docs/execute_api/auto_data_types):** Turn your database into a centralized repository for storing and managing inputs and outputs of AI models of arbitrary data-types, making them available in a structured format and known environment.
- **[Vector search](https://docs.superduper.io/docs/tutorials/vector_search):** No need to duplicate and migrate your data to additional specialized vector databases - turn your existing battle-tested database into a fully-fledged multi-modal vector-search database, including easy generation of vector embeddings and vector indexes of your data with preferred models and APIs.

## Preview

[Browse the re-usable snippets](https://docs.superduper.io/docs/category/reusable-snippets) to understand how to accomplish difficult AI end-functionality
with few lines of code using Superduper.



## Example use-cases and apps (notebooks)
The notebooks below are examples how to make use of different frameworks, model providers, databases, retrieval techniques and more. To learn more about *how* to use Superduper with your database, please check our [Docs](https://docs.superduper.io).

<table >

| Name                                            | Link                                                                                                                                              |
|-------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| Multimodal vector-search with a range of models and datatypes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/superduper-io/superduper/blob/main/docs/content/use_cases/multimodal_vector_search_image.ipynb) |
| RAG with self-hosted LLM                        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/superduper-io/superduper/blob/main/docs/content/use_cases/retrieval_augmented_generation.ipynb)                     |
| Fine-tune an LLM on your database               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/superduper-io/superduper/blob/main/docs/content/use_cases/fine_tune_llm_on_database.ipynb)                        |
| Featurization and transfer learning             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/superduper-io/superduper/blob/main/docs/content/use_cases/transfer_learning.ipynb)                               |

</table >
</div>


## Currently supported datastores:
Superduper your: [MongoDB](https://www.mongodb.com),
[MongoDB Atlas](https://www.mongodb.com/cloud/atlas),
[Snowflake](https://www.snowflake.com),
[PostgreSQL](https://www.postgresql.org), 
[MySQL](https://www.mysql.com),
[SQLite](https://www.sqlite.org),
[DuckDB](https://duckdb.org),
[Google BigQuery](https://cloud.google.com/bigquery),
[Amazon S3](https://aws.amazon.com/s3/),
[Microsoft SQL Server (MSSQL)](https://www.microsoft.com/en-us/sql-server),
[ClickHouse](https://clickhouse.com),
[Oracle](https://www.oracle.com/database/),
[Trino](https://trino.io),
[PySpark](https://spark.apache.org/docs/latest/api/python/),
[Pandas](https://pandas.pydata.org),
[Apache Druid](https://druid.apache.org),
[Apache Impala](https://impala.apache.org),
[Polars](https://www.pola.rs),
[Apache Arrow DataFusion](https://arrow.apache.org/datafusion/),



## Supported AI frameworks, models and APIs (*more coming soon*):

Integrate and self-hosted your own models (whether from open-source, commercial or self-developed) with a simple Python command from: [PyTorch](https://pytorch.org), [Scikit-learn](https://scikit-learn.org), [HuggingFace](https://huggingface.co) 


## Preconfigured API integrations (*more coming soon*):

Integrate externally hosted models accessible via API to work side-by-side or together with your other models a simple Python command: [OpenAI](https://www.openai.com), [Cohere](https://cohere.ai), [Anthropic](https://www.anthropic.com), [Jina AI](https://jina.ai)


## Installation

#### # Option 1. Superduper Library
Ideal for building new AI applications.
```shell
pip install superduper-framework
```

#### # Option 2. Superduper Container
Ideal for learning basic Superduper functionalities and testing notebooks.
```shell
docker pull superduperio/superduper
docker run -p 8888:8888 superduperio/superduper
```

#### # Option 3. Superduper Testenv
Ideal for learning advanced Superduper functionalities and testing whole AI stacks.
```shell
make build_sandbox
make testenv_init
```


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
