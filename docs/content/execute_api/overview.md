# Execute API

SuperDuperDB implements 2 main classes of `db.databackend`:

- [MongoDB](../data_integrations/mongodb)
- [SQL backends](../data_integrations/sql)

Correspondingly, SuperDuperDB currently has 2 flavours of query API:

- [`pymongo`](https://pymongo.readthedocs.io/en/stable/)
- [`ibis`](https://ibis-project.org/)

## Base

A few commands are shared in common by all supported databackends:

- `db["table_name"].insert(data)`
- `db["table_name"].select()`

For more specific commands, one should use one of the two following APIs.

## PyMongo

`pymongo` is the official MongoDB client for Python. It supports 
compositional queries, leveraging the BSON format for encoding 
and retrieving data.

## Ibis

`ibis` is a Python library with a uniform compositional approach to building
SQL queries.