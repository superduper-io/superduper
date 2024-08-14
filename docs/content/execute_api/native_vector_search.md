# Using a database's native vector search

Databases increasingly include their own vector-comparison and search operations 
(comparing one vector with others). In order to use this, include 
this configuration in your configuration setup:

```yaml
cluster:
  vector_search:
    type: native
```

***or***

```bash
export SUPERDUPER_CLUSTER_VECTOR_SEARCH_TYPE=native
```

If `superduper` detects this configuration, it uses the inbuilt mechanism 
of your `db.databackend` to perform the vector-comparison.

Currently Superduper supports the native implementation of these databases:

- MongoDB Atlas

More integrations are on the way.