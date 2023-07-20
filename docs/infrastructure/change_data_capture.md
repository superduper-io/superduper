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

Correspondingly, 

DatabaseWatcher for MongoDB in Python

The DatabaseWatcher class is a Python implementation of a Change Data Capture (CDC) solution for MongoDB. It allows you to monitor changes in a specified collection by utilizing a daemon thread that watches the change stream.


## Usage


Import the necessary dependencies:
```python
from superduperdb.datalayer.mongodb.cdc import DatabaseWatcher
```

Instantiate a DatabaseWatcher object by providing the MongoDB database and the collection to be monitored:

```python
watcher = DatabaseWatcher(db=db, on=Collection(name='docs'))
```

Start the watcher thread to initiate the change stream monitoring:
```python
watcher.watch()
```

## `DatabaseWatcher`

Example:

```python
watcher = DatabaseWatcher(db=db, on=Collection(name='docs'))
```

### `.watch`

Starts the watcher thread and initiates the change stream monitoring.

```python
watcher.watch()
```

This method creates and starts a daemon thread that continuously listens to changes in the specified MongoDB collection.

### `.is_available`

Gives the current avaiblitity of the watcher.

```python
watcher.is_available()
```
### `.info`
Gets the current dictionary info of database watcher

```python
watcher.info()
```
Returns a dictionary with inserts and updates encountered.

Detailed architecture of the cdc process.

```{mermaid}
    graph LR
        A[CDC thread] -- sends changed events --> B((Queue))
        B --> D[CDC event handler]
```

