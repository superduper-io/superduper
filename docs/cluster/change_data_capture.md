(CDC)=
# Change data capture

SuperDuperDB has been designed from the ground up to provide first class
support for keeping AI predictions up-to-date with database changes.

If MongoDB is deployed in standalone mode, then changes are detected
by the same process as data is inserted, resp., updated and deleted by.

For deployments involving multiple stakeholders, and roles, this may be 
overly restrictive. Data inserts and updates may occur from:

- CI/ CD processes triggering data ingestion to MongoDB
- Ingestion/ updates from non-SuperDuperDB client libraries:
  - [`pymongo`](https://pymongo.readthedocs.io/en/stable/)
  - The MongoDB shell: [`mongo`](https://www.mongodb.com/docs/v4.4/mongo/)
  - Client libraries from non-Python programming languages

SuperDuperDB aims to allow it's AI models to be updated and kept in-sync with changes
from all of the above sources.

Correspondingly, the `DatabaseListener` class is a Python implementation of a Change Data Capture (CDC) solution for MongoDB. It allows you to monitor changes in a specified collection by utilizing a daemon thread that listens to the change stream.

## Usage

Import the necessary dependencies:

```python
from superduperdb.db.mongodb.cdc import DatabaseListener
```

Instantiate a `DatabaseListener` object by providing the MongoDB database and the collection to be monitored:

```python
listener = DatabaseListener(db=db, on=Collection(name='docs'))
```

Start the listener thread to initiate the change stream monitoring:
```python
listener.listen()
```

See [here](/how_to/mongo_cdc.html) for an example of usage of CDC.
