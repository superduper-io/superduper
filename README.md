<a href="https://www.superduperdb.com"><img src="https://raw.githubusercontent.com/blythed/superduperdb/main/img/symbol_purple.png" width="150" align="right" /></a>

# Welcome to SuperDuperDB!

> An AI-database management system for the full PyTorch model-development lifecycle

Full documentation [here](https://superduperdb.github.io/superduperdb).

## Installation

Requires:

- MongoDB
- RedisDB

Then install the python requirements

```
pip install -r requirements.txt
```

## Contributing

SuperDuperDB is in *alpha*. Please expect:

- breaking changes
- rough edges
- fast pace of new feature development

If you'd like to contribute to the project we need help in many places:

- Trying out the features and commenting on the issues boards
- Adding additional unittests and doctests
- Augmenting doc-strings to make the usage patterns clearer for the uninitiated
- Expanding the documentation, tutorials and examples

## Use Cases

*Vector search*

1. OpenAI vector search
1. PineCone vector search
1. MongoDB atlas
1. ElasticSearch

Saving your own model

1. Bert
1. CLIP
1. Other (e.g. audio)

Support for [cohere.ai]()

1. Compare with OpenAI
1. Arbitrary API

Llamaindex

Reimplement entire use-case:
https://gpt-index.readthedocs.io/en/latest/guides/tutorials/building_a_chatbot.html

Unstructured

Dump of pdfs and apply unstructured.io or arxiv.org pdfs

Potentially build examples with retool

https://retool.com/?_keyword=retool&adgroupid=133907531011&utm_source=google&utm_medium=search&utm_campaign=14901002285&utm_term=retool&utm_content=651513675598&hsa_acc=7420316652&hsa_cam=14901002285&hsa_grp=133907531011&hsa_ad=651513675598&hsa_src=g&hsa_tgt=kwd-395242915847&hsa_kw=retool&hsa_mt=e&hsa_net=adwords&hsa_ver=3&gad=1&gclid=CjwKCAjwge2iBhBBEiwAfXDBR17UdmTipws28yPI3vINR2YLcnnX1ln6oahaHc_T6yOBWAvP7Wx1BRoClCwQAvD_BwE

RSS news feeds

Talk to these in real time
Zapier or something

API to atlassian/ notion/ TBD

...

Zoom meetings transcription + ChatBot

...

Time-series analysis with Sktime

...


## Project organization

```
|_models
|_providers
| |_openai
| |_cohereai
|_apps
| |_langchain
| |_llamaindex
|_vectorsearch
|_...
```

```