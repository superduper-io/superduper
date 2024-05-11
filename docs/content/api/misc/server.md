**`superduperdb.misc.server`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/misc/server.py)

## `request_server` 

```python
request_server(service: str = 'vector_search',
     data=None,
     endpoint='add',
     args={},
     type='post')
```
| Parameter | Description |
|-----------|-------------|
| service | Service name |
| data | Data to send |
| endpoint | Endpoint to hit |
| args | Arguments to pass |
| type | Type of request |

Request server with data.

## `server_request_decoder` 

```python
server_request_decoder(x)
```
| Parameter | Description |
|-----------|-------------|
| x | Object to decode. |

Decodes a request to `SuperDuperApp` service.

