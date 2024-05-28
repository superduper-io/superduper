# RESTful service

## Start the server

Start a development version of the service on port 8002 with:

```bash
superduperdb rest
```

To change the port and ip-range allowed configure:

```yaml
cluster:
  rest:
    uri: http://x1.x2.x3.x4:port
    config: local/path/to/config.yaml  # optional
```

Read more about configuration [here](../core_api/connect) and [here](../connect_api).

In principle, once connected, it is possible to everything with this REST API as can be achieved
from Python.

## Important Endpoints

| Route | Method | Data | Params |
| --- | --- | --- | --- |
| `/db/apply` | `POST` | [JSON defining `Component`](./yaml_formalism)| N/A | 
| `/db/execute` | `POST` | `{"query": <str>, "documents": <list_documents>}` | N/A |
| `/db/remove` | `POST` | `{"type_id": <type_id>, "identifier": <identifier>}` | N/A |
| `/db/show` | `GET` | N/A | `?type_id=<type_id>` |
| `/db/artifact_store/put` | `PUT` | `<file>` | `?datatype=<datatype>` |
| `/db/artifact_store/get` | `GET` | N/A | `?file_id=<file_id>` |