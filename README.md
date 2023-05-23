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

## Development

### Running Tests

Our test suite relies on `docker` to run 3rd-party services.

```shell
make test
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

RSS news feeds

Talk to these in real time
Zapier or something

API to atlassian/ notion/ TBD

...

Zoom meetings transcription + ChatBot

...

Time-series analysis with Sktime

...


