<p align="center">
  <a href="https://www.superduper.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/img/SuperDuperDB_logo_white.svg">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/img/SuperDuperDB_logo_color.svg">
      <img width="50%" alt="SuperDuper logo" src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/img/SuperDuperDB_logo_color.svg">
    </picture>
  </a>
</p>

<div align="center">
  <h2>Bring AI to your favorite database!</h2>
  <p>
    <a href="https://docs.superduper.io"><strong>Docs</strong></a> |
    <a href="https://blog.superduper.io"><strong>Blog</strong></a> |
    <a href="https://docs.superduper.io/docs/category/use-cases"><strong>Use-Cases</strong></a> |
    <a href="https://docs.superduper.io/docs/docs/get_started/installation"><strong>Installation</strong></a> |
    <a href="https://github.com/superduper-io/superduper-community-apps"><strong>Community Apps</strong></a> |
    <a href="https://join.slack.com/t/superduper/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA"><strong>Slack</strong></a> |
    <a href="https://www.youtube.com/channel/UC-clq9x8EGtQc6MHW0GF73g"><strong>Youtube</strong></a>
  </p>

  <p>
    <a href="https://pypi.org/project/superduper-framework"><img src="https://img.shields.io/pypi/v/superduper-framework?color=%23007ec6&label=pypi%20package" alt="Package version"></a>
    <a href="https://pypi.org/project/superduper-framework"><img src="https://img.shields.io/pypi/pyversions/superduper-framework" alt="Supported Python versions"></a>    
    <a href="https://github.com/superduper-io/superduper/actions/workflows/ci_code.yml"><img src="https://github.com/superduper-io/superduper/actions/workflows/ci_code.yml/badge.svg?branch=main" /></a>
    <a href="https://github.com/superduper-io/superduper/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-green" alt="License - Apache 2.0"></a>	
    <a href="https://twitter.com/superduperdb" target="_blank"><img src="https://img.shields.io/twitter/follow/nestframework.svg?style=social&label=Follow @superduper.io"></a>
  </p>
</div>

---

### :mega: We are rebranding from [SuperDuperDB](https://github.com/SuperDuperDB/superduperdb) to [superduper-io](https://github.com/superduper-io/superduper). Please bear with us during this transition!

---

## What is superduper.io? 🔮 

`superduper.io` is a Python framework for integrating AI models, APIs, and vector search engines directly with your existing databases. It includes hosting of your own models, streaming inference, and scalable model training/fine-tuning.

<p align="center">
  <img width="70%" src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/img/superduper.gif">
</p>

## Key Features


| Feature | Description |
|---------|-------------|
| **[Seamless AI Integration with Existing Data Infrastructure](https://docs.superduper.io/docs/docs/walkthrough/apply_models)** | Effortlessly integrate AI models and APIs with your current databases, without requiring additional pre-processing steps, ETL, or boilerplate code. |
| **[ Real-time Inference via Change-Data-Capture](https://docs.superduper.io/docs/docs/walkthrough/daemonizing_models_with_listeners)** | Enable automatic computation of outputs as new data is ingested, ensuring your AI deployment remains current and responsive. |
| **[Scalable Model Training on Large Datasets](https://docs.superduper.io/docs/docs/walkthrough/training_models)** | Facilitate the training of AI models on extensive datasets by querying your data with integrated computational optimizations, ensuring efficient and scalable model training processes. |
| **[Complex Model Chaining](https://docs.superduper.io/docs/docs/walkthrough/linking_interdependent_models)** | Establish intricate workflows by connecting multiple models and APIs in a dependent sequence, enabling sophisticated data processing and decision-making pipelines. |
| **[Friendly Python Interface](https://docs.superduper.io/docs/docs/fundamentals/procedural_vs_declarative_api)** | Minimize the need for extensive glue code by utilizing simple Python commands while retaining access to detailed implementation layers for advanced customization. |
| **[Python-First Approach](https://docs.superduper.io/docs/docs/fundamentals/procedural_vs_declarative_api)** | Leverage the full potential of the Python ecosystem, incorporating any function, script, or algorithm to enhance and extend your AI workflows seamlessly. |
| **[Support for Complex Data Types](https://docs.superduper.io/docs/docs/walkthrough/encoding_special_data_types)** | Directly work with a variety of data types, including images, video, audio, and byte-encoded data, within Python, ensuring comprehensive data handling capabilities. |
| **[AI Feature Storage](https://docs.superduper.io/docs/docs/walkthrough/encoding_special_data_types)** | Transform your database into a structured repository for managing AI model inputs and outputs, facilitating efficient tracking and management of feature data. |
| **[Multi-Modal Vector Search](https://docs.superduper.io/docs/docs/walkthrough/vector_search)** | Convert your database into a sophisticated multi-modal vector-search engine, generating and indexing vector embeddings with your chosen models and APIs for advanced search capabilities. |



<div align="center">
  <a href="https://www.youtube.com/watch?v=dDJ_ktMtbw0"><img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/img/video.png" alt="How to get Started" width="500"></a>
</div>


For more features, feel free to [contribute to our Roadmap for `v0.3`](https://github.com/superduper-io/superduper/discussions/1882).


## Example Use-Cases and Apps (Notebooks)

[Explore the reusable snippets](https://docs.superduper.io/docs/category/reusable-snippets) of superduper.io to learn how to achieve complex AI functionality with just a few lines of code. The following notebooks showcase different frameworks, model providers, vector databases, and retrieval techniques.

<table>
  <tr>
    <th>Name</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>Multimodal vector-search with various models and datatypes</td>
    <td><a href="https://colab.research.google.com/github/superduper-io/superduper/blob/main/docs/content/use_cases/multimodal_vector_search_image.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td>
  </tr>
  <tr>
    <td>RAG with self-hosted LLM</td>
    <td><a href="https://colab.research.google.com/github/superduper.io/superduper/blob/main/docs/content/use_cases/retrieval_augmented_generation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td>
  </tr>
  <tr>
    <td>Fine-tune an LLM on your database</td>
    <td><a href="https://github.com/superduper-io/superduper/blob/main/docs/content/use_cases/fine_tune_llm_on_database.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td>
  </tr>
  <tr>
    <td>Featurization and transfer learning</td>
    <td><a href="https://github.com/superduper-io/superduper/blob/main/docs/content/use_cases/transfer_learning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td>
  </tr>
</table>

For more information, check our [Docs](https://docs.superduper.io/), our [Tutorials](https://docs.superduper.io/docs/docs/walkthrough/tutorial_walkthrough), and [read this blog post](https://blog.superduper.io/superduper-the-open-source-framework-for-bringing-ai-to-your-datastore/).


## Installation

#### # Option 1. superduper.io Library
Ideal for building new AI applications.
```shell
pip install superduper-framework
```

#### # Option 2. superduper.io Container
Ideal for learning basic superduper.io functionalities and testing notebooks.
```shell
docker pull superduperio/superduper
docker run -p 8888:8888 superduperio/superduper
```

#### # Option 3. superduper.io Testenv
Ideal for learning advanced superduper.io functionalities and testing whole AI stacks.
```shell
make build_sandbox
make testenv_init
```

## Current Integrations 

####  Datastores

<table>
    <tr>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/databases/2Xmongodb.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/databases/2Xmongodb-atlas.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/databases/2Xaws-s3.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/databases/2Xpostgresql.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/databases/2Xmysql.png" width="139px" />
        </td>
    </tr>
    <tr>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/databases/2Xsqlite.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/databases/2Xduckdb.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/databases/2Xsnowflake.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/databases/2Xbigquery.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/databases/2Xclickhouse.png" width="139px" />
        </td>
    </tr>
    <tr>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/databases/2Xmssql.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/databases/2Xoracle.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/databases/2Xpandas.png" width="139px" />
        </td>	    
    </tr>
</table>


#### AI Frameworks and Models

<table>
    <tr>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/frameworks/2Xpytorch.png" width="139px"/>
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/frameworks/2Xscikit-learn.png" width="139px"/>
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/frameworks/2Xhuggingface-transformers.png" width="139px"/>
        </td>
    </tr>
</table>


#### AI APIs

<table >
    <tr>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/apis/2Xopenai.png" width="139px"/>
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/apis/2Xcohere.png" width="139px"/>
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/apis/2Xanthropic.png" width="139px"/>
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/superduper-io/superduper/main/docs/static/icons/apis/jinaai.png" width="139px"/>
        </td>
    </tr>
</table>


For more integrations, feel free to [contribute to our Roadmap for `v0.3`](https://github.com/superduper-io/superduper/discussions/1882).



## Community & Getting Help 

#### If you have any problems, questions, comments, or ideas:
- Join <a href="https://join.slack.com/t/superduper/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA">our Slack</a> (we look forward to seeing you there).
- Search through <a href="https://github.com/superduper-io/superduper/discussions">our GitHub Discussions</a>, or <a href="https://github.com/superduper-io/superduper/discussions/new/choose">add a new question</a>.
- Comment <a href="https://github.com/superduper-io/superduper/issues/">an existing issue</a> or create <a href="https://github.com/superduper-io/superduper/issues/new/choose">a new one</a>.
- Help us to improve superduper.io by providing your valuable feedback <a href="https://docs.google.com/forms/d/e/1FAIpQLScKNyLCjSEWAwc-THjC7NJVDNRxQmVR5ey30VVayPhWOIhy1Q/viewform">here</a>!
- Email us at `gethelp@superduper.io`.
- Feel free to contact a maintainer or community volunteer directly! 



## Contributing  

#### There are many ways to contribute, and they are not limited to writing code. We welcome all contributions such as:


- <a href="https://github.com/superduper-io/superduper/issues/new/choose">Bug reports</a>
- <a href="https://github.com/superduper-io/superduper/issues/new/choose">Documentation improvements</a>
- <a href="https://github.com/superduper-io/superduper/issues/new/choose">Enhancement suggestions</a>
- <a href="https://github.com/superduper-io/superduper/issues/new/choose">Feature requests</a>
- <a href="https://github.com/superduper-io/superduper/issues/new/choose">Expanding the tutorials and use case examples</a>

Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Contributors
#### Thanks goes to these wonderful people:

<a href="https://github.com/superduper-io/superduper/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=superduperdb/superduper" />
</a>


## License  

superduper.io is open-source and intended to be a community effort, and it wouldn't be possible without your support and enthusiasm.
It is distributed under the terms of the Apache 2.0 license. Any contribution made to this project will be subject to the same provisions.

## Join Us 

We are looking for nice people who are invested in the problem we are trying to solve to join us full-time. Find roles that we are trying to fill <a href="https://join.com/companies/superduper">here</a>!

