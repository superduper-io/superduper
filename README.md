<a href="https://www.superduperdb.com">
  <img
    src="img/symbol_purple.png"
    width="150"
    align="right"
    />
</a>

# Welcome to SuperDuperDB!

> An AI-database management system for the full PyTorch model-development lifecycle

Full documentation [here](https://superduperdb.github.io/superduperdb).

## Installation

Requires:

- MongoDB
- RedisDB
- poetry

Then install the python requirements

```
poetry install
```

## Running the CLI

To run the superduperdb cli, from the root directory of the project, type

```
python -m superduperdb
```

#### Examples:

```
python -m superduperdb -h       # Get help
python -m superduperdb configs  # Print current configuration variables
python -m superduperdb info     # Print info on platform and installation
python -m superduperdb test     # Run tests
```

## Development

### Running Tests

Our test suite relies on `docker` to run 3rd-party services.

```shell
make test
```

### Running Jupyter

You might want to run Jupyter equipped with SuperDuperDB Client and connected to our
various components.
The notebooks in the `./notebooks` directory are automatically mounted and available
within your Jupyter instance.

```shell
make jupyter
make clean-jupyter
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

## Using configs

SuperDuperDB has "config variables" that can be set to customize its operation.

You can see a list of their default values in this file https://github.com/SuperDuperDB/superduperdb-stealth/blob/main/default-configs.json.

In the code, configs are simple data classes, defined here: https://github.com/SuperDuperDB/superduperdb-stealth/blob/main/superduperdb/misc/config.py

There are three ways to set a config variable

* put just the values you want to change in a file `configs.json` at the room of `superduperdb-stealth`
* set an environment variable with the value
* set it in code

For example, these three forms are identical:

* Storing `{"remote": True, "dask": {"ip": "1.1.1.1"}}` in `configs.json`
* Setting environment variables `SUPERDUPERDB_REMOTE=true` and
  `SUPERDUPERDB_DASK_IP=1.1.1.1`
* In Python, `CFG.remote = True; CFG.dask.ip = '1.1.1.1'`

[TBD: Secrets are just the same, except we don't even have one yet.]


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
