# Query API


SuperDuperDB implements 2 main classes of `db.databackend`:

- [MongoDB](../data_integrations/mongodb)
- [SQL backends](../data_integrations/sql)

Correspondingly, SuperDuperDB currently has 2 flavours of query API:

- `pymongo`
- `ibis`

## PyMongo

`pymongo` is the official MongoDB client for Python. It supports 
compositional queries, leveraging the BSON format for encoding 
and retrieving data.

## Ibis

`ibis` is a Python library with a uniform compositional approach to building
SQL queries. It communicates with the `db.databackend` using `pandas`.