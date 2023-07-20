# Supported Features

This is a feature list of features supported in `v0.1`.

See [here](https://github.com/SuperDuperDB/superduperdb-stealth/wiki/Roadmap) for the high-level roadmap going forward.

## Components

- *Models*
- *Encoders*
- *Watchers*
- *Metrics*
- *Vector-Indexes*
- *Datasets*

## AI Models

### API structure

All model frameworks are equipped with:

- `model.predict(X, db=<db>, select=<query>)`

Optional support for:

- `model.fit(X, y=None, db=<db>, select=<query>)`

### In-database compute

For `model.fit` and `model.predict`:

1. All model frameworks may applied client-side, automatically fetching data from the data layer, and iterating through this data, reinserting predictions into the database
2. The same workflow may be flagged to happen "in" the SuperDuperDB environment.
3. Workflows may be configured to run in parallel on Dask workers

### Supported and tested AI frameworks

- **PyTorch**
- **Sklearn**
- Hugging Face's **Transformers**
- **Sentence-Transformers**
- **OpenAI**
- **LangChain**

### Versioning and storage

Models, and components associated with models are automatically stored in a configured artifact store, and versioned. Dependencies between models and components are logged, and versions are protected from deletion which participate in other components.

### Watchers: daemonizing models on datalayer

Models may be configured to watch the database for changes, and when new data is inserted, compute new outputs over that data, and store this back in the datalayer.

## Datalayer

### Supported datalayers

Currently supported

- **MongoDB**

View plans to support additional databases in the roadmap [here](https://github.com/SuperDuperDB/superduperdb-stealth/wiki/Roadmap).

### Datatypes

Native support:

- **images**
- **tensors**
- **arrays**

Configurable

- Any data encodable using a saveable *Encoder* with `.encode` and `.decode` methods

### Model Feature storage

The outputs of models may be stored directly in the database along-side the inputs which led to those outputs. These outputs may be flexibly be of any type which can be handled by a SuperDuperDB *Encoder*.

## Features for production use-cases

We have support for these features:

- Task parallelization on Dask
- Client-server for working with SuperDuperDB from a remote client
- Change-data capture (CDC): users are enabled to insert data from any client and models may be executed on workers, capturing new inputs and populating outputs to the datalayer
- Vector-search using LanceDB, which is kept in sync with the datalayer using CDC.