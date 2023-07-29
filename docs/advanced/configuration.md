# Configuration

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
    "distributed": false,
    "cdc": false,
    "server": {
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
