# Quick  start

## Prerequisites

We assume you are running:

- Python3.8+
- MongoDB

The deployments can be existing deployments already running in your infrastructure, or
bespoke deployments which you'll use specifically for SuperDuperDB.

For MongoDB here are instructions for 2 very popular systems:

- [MongoDB installation on Ubuntu](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu)
- [MongoDB installation on MacOSX](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-os-x/)

## Installation

`pip install superduperdb`

## Configuration

`superduperdb/misc/config.py` contains a complete definition using `pydantic` of the default
configuration. This may be overridden using a `configs.json` file containing in the 
working directory. 

There are three ways to set a config variable

* put just the values you want to change in a file `configs.json` at the room of `superduperdb-stealth`
* set an environment variable with the value
* set it in code

For example, these three forms are identical:

* Storing `{"remote": True, "dask": {"ip": "1.1.1.1"}}` in `configs.json`
* Setting environment variables `SUPERDUPERDB_REMOTE=true` and
  `SUPERDUPERDB_DASK_IP=1.1.1.1`
* In Python, `CFG.remote = True; CFG.dask.ip = '1.1.1.1'`

Here is what the default config object looks like in `JSON` format:

```json
{
    "apis": {
        "providers": {},
        "retry": {
            "wait_multiplier": 1.0,
            "wait_min": 4.0,
            "wait_max": 10.0,
            "stop_after_attempt": 2
        }
    },
    "dask": {
        "port": 8786,
        "password": "",
        "username": "",
        "ip": "localhost",
        "serializers": [],
        "deserializers": []
    },
    "logging": {
        "kwargs": {},
        "level": "INFO",
        "type": "STDERR"
    },
    "data_layers": {
        "artifact": {
            "cls": "mongodb",
            "connection": "pymongo",
            "kwargs": {
                "port": 27017,
                "password": "",
                "username": "",
                "host": "localhost"
            },
            "name": "_filesystem:documents"
        },
        "data_backend": {
            "cls": "mongodb",
            "connection": "pymongo",
            "kwargs": {
                "port": 27017,
                "password": "",
                "username": "",
                "host": "localhost"
            },
            "name": "documents"
        },
        "metadata": {
            "cls": "mongodb",
            "connection": "pymongo",
            "kwargs": {
                "port": 27017,
                "password": "",
                "username": "",
                "host": "localhost"
            },
            "name": "documents"
        }
    },
    "notebook": {
        "ip": "0.0.0.0",
        "port": 8888,
        "password": "",
        "token": ""
    },
    "ray": {
        "port": 0,
        "password": "",
        "username": "",
        "host": "127.0.0.1",
        "deployments": []
    },
    "remote": false,
    "cdc": false,
    "server": {
        "fastapi": {
            "debug": false,
            "title": "SuperDuperDB server",
            "version": "0.1.0"
        },
        "web_server": {
            "host": "127.0.0.1",
            "port": 3223,
            "protocol": "http"
        },
        "test_port": 32233
    },
    "vector_search": {
        "lancedb": {
            "uri": "./.lancedb"
        }
    }
}
```

## Command line interface

The `superduperdb` package ships with a command line interface (CLI), in order to perform
important tasks such as:

**Starting the server**

...

**Printing out the current configuration**

...

**Printing out important information relating to the current setup**

- Encoders
- Models
- Watchers
- VectorIndexes
- Jobs

## Minimum working example

To check that everything is working correctly try the notebook "minimum-working-example.ipynb"
in the `notebooks/` directory. For completeness, here is the code to execute:

```python
import numpy as np
from pymongo import MongoClient
from superduperdb.core.documents import Document as D
from superduperdb.encoders.numpy.array import array
from superduperdb.datalayer.mongodb.query import Collection, InsertMany
import superduperdb as s

db = s.superduper(MongoClient().documents)
collection = Collection(name='docs')

a = array('float64', shape=(32,))

db.execute(
    collection.insert_many([
        D({'x': a(np.random.randn(32))})
        for _ in range(100)
    ], encoders=(a,))
)

print(db.execute(collection.find_one()))
# prints:

```

## Cluster Setup

To run SuperDuperDB in asynchronous mode, one is required to set up a SuperDuperDB cluster.

...
