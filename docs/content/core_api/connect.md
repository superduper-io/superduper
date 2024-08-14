# Connect

The standard way to connect to Superduper is via the `superduper` decorator:

## Development mode

In [development mode](../get_started/environment#development-mode), you may provide the URI/ connection details of your data deployment directly

```python
db = superduper('<database-uri>')
```

For example if you are running a (not secure) MongoDB deployment locally, and you want to connect to the `"documents"` database, you might write:

```python
from superduper import superduper
db = superduper('mongodb://localhost:27017/documents')
```

### Complete connection guide

For a semi-exhaustive list of possible connections see [here](../reusable_snippets/connect_to_superduper).

### Fine grained configuration

`superduper` chooses default `artifact_store` (file blob storage) and `metadata_store` (AI metadata) values for your connection. These defaults may be overridden directly:

```python
db = superduper(
    '<database-uri>',
    artifact_store='<artifact-store-uri>,
    metadata_store='<metadata-store-uri>,
)
```

## Cluster mode

In [cluster mode](../get_started/environment#cluster-mode), the connection string needs to be provided in a configuration 
file or via environment variable so that all services can connect correctly:

Add these lines to your configuration:

```yaml
data_backend: mongodb://localhost:27018/documents
```

Read more about configuration [here](../get_started/configuration).

After doing this, you may connect directly with the `superduper` decorator:

```python
db = superduper()
```

### Fine grained configuration

For more granular configuration add these lines:


```yaml
data_backend: <database-uri>,
artifact_store: <artifact-store-uri>
metadata_store: <metadata-store-uri>
```

## Next steps

`db` is now your connection to your data, models, and model meta-data.
Now that you have established this connection you are ready to build, deploy and manage AI with Superduper.
