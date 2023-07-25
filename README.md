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
<a href="https://colab.research.google.com/drive/11SJunSZc52jUuYrmNi5GziombcQ_sdhJ#scrollTo=XwWu32JBovja">
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

- **Deploy/ Install a Pytorch, sklearn or HuggingFace model <a href="404" target="_blank">(read more in the docs here)</a>:**
```python
import superduperdb
```
- **Train/ fine-tune a model <a href="404" target="_blank">(read more in the docs here)</a>:**
```python
import superduperdb
```

- **Create downstream classifier model <a href="404" target="_blank">(read more in the docs here)</a>:**
```python
install llama2
```

- **Use MongoDB as your vector search database <a href="404" target="_blank">(read more in the docs here)</a>:**
```python
install llama2
```

- **Integrate externally hosted models gated via an API (such as OpenAI) <a href="404" target="_blank">(read more in the docs here)</a>:**
```python
import superduperdb
```
- **Integrate LangChain <a href="404" target="_blank">(read more in the docs here)</a>:**
```python
install llama2
```
- **Integrate Llama 2 as a HuggingFace transformer <a href="404" target="_blank">(read more in the docs here)</a>:**
```python
install llama2
```

- **Create downstream classifier model <a href="404" target="_blank">(read more in the docs here)</a>:**
```python
install llama2
```

# Installation :electric_plug:

**1. Install SuperDuperDB via `pip *(~1 minute)*:**
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
