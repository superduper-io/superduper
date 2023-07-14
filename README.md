<a href="https://www.superduperdb.com">
  <img
    src="https://raw.githubusercontent.com/superduperdb/superduperdb-stealth/img/symbol_purple.png"
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

### Running Tests

Our test suite relies on `docker` to run 3rd-party services.

```shell
make test
```

After the first time you can just run `pytest`.

If you want to extract annotations automatically, use `pytest
--annotate-output=./annotations.json`.

## Using configs

SuperDuperDB has "config variables" that can be set to customize its operation.

In the code, configs are simple data classes, defined [here](https://github.com/SuperDuperDB/superduperdb-stealth/blob/main/superduperdb/misc/config.py).

There are three ways to set a config variable

* put just the values you want to change in a file `configs.json` at the room of `superduperdb-stealth`
* set an environment variable with the value
* set it in code

For example, these three forms are identical:

* Storing `{"remote": True, "dask": {"ip": "1.1.1.1"}}` in `configs.json`
* Setting environment variables `SUPERDUPERDB_REMOTE=true` and
  `SUPERDUPERDB_DASK_IP=1.1.1.1`
* In Python, `CFG.remote = True; CFG.dask.ip = '1.1.1.1'`

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

