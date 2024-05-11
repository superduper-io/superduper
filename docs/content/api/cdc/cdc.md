**`superduperdb.cdc.cdc`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/cdc/cdc.py)

## `DatabaseChangeDataCapture` 

```python
DatabaseChangeDataCapture(self,
     db: 'Datalayer')
```
| Parameter | Description |
|-----------|-------------|
| db | A superduperdb datalayer instance. |

DatabaseChangeDataCapture (CDC).

DatabaseChangeDataCapture is a Python class that provides a flexible and
extensible framework for capturing and managing data changes
in a database.

This class is repsonsible for cdc service on the provided `db` instance
This class is designed to simplify the process of tracking changes
to database records,allowing you to monitor and respond to
data modifications efficiently.

## `BaseDatabaseListener` 

```python
BaseDatabaseListener(self,
     db: 'Datalayer',
     on: Union[ForwardRef('IbisQuery'),
     ForwardRef('TableOrCollection')],
     stop_event: superduperdb.misc.runnable.runnable.Event,
     identifier: 'str' = '',
     timeout: Optional[float] = None)
```
| Parameter | Description |
|-----------|-------------|
| db | A superduperdb instance. |
| on | A table or collection on which the listener is invoked. |
| stop_event | A threading event flag to notify for stoppage. |
| identifier | A identity given to the listener service. |
| timeout | A timeout for the listener. |

A Base class which defines basic functions to implement.

This class is responsible for defining the basic functions
that needs to be implemented by the database listener.

## `CDCHandler` 

```python
CDCHandler(self,
     db: 'Datalayer',
     stop_event: superduperdb.misc.runnable.runnable.Event,
     queue)
```
| Parameter | Description |
|-----------|-------------|
| db | A superduperdb instance. |
| stop_event | A threading event flag to notify for stoppage. |
| queue | A queue to hold the cdc packets. |

CDCHandler for handling CDC changes.

This class is responsible for handling the change by executing the taskflow.
This class also extends the task graph by adding funcation job node which
does post model executiong jobs, i.e `copy_vectors`.

## `DatabaseListenerFactory` 

```python
DatabaseListenerFactory(self,
     db_type: str = 'mongodb')
```
| Parameter | Description |
|-----------|-------------|
| db_type | Database type. |

DatabaseListenerFactory to create listeners for different databases.

This class is responsible for creating a DatabaseListener instance
based on the database type.

## `DatabaseListenerThreadScheduler` 

```python
DatabaseListenerThreadScheduler(self,
     listener: superduperdb.cdc.cdc.BaseDatabaseListener,
     stop_event: superduperdb.misc.runnable.runnable.Event,
     start_event: superduperdb.misc.runnable.runnable.Event) -> None
```
| Parameter | Description |
|-----------|-------------|
| listener | A BaseDatabaseListener instance. |
| stop_event | A threading event flag to notify for stoppage. |
| start_event | A threading event flag to notify for start. |

DatabaseListenerThreadScheduler to listen to the cdc changes.

This class is responsible for listening to the cdc changes and
executing the following job.

## `Packet` 

```python
Packet(self,
     ids: Any,
     query: Optional[Any] = None,
     event_type: superduperdb.cdc.cdc.DBEvent = <DBEvent.insert: 'insert'>) -> None
```
| Parameter | Description |
|-----------|-------------|
| ids | Document ids. |
| query | Query to fetch the document. |
| event_type | CDC event type. |

Packet to hold the cdc event data.

