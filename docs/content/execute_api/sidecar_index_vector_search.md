# Sidecar vector-comparison integration

For databases which don't have their own vector-search implementation, `superduperdb` offers 
2 integrations:

- In memory vector-search
- Lance vector-search

To configure these, add one of the following options to your configuration:

```yaml
cluster:
  vector_search:
    type: in_memory|lance
```

***or***

```bash
export SUPERDUPER_CLUSTER_VECTOR_SEARCH_TYPE='in_memory|lance'
```

In this case, whenever a developer executes a vector-search query including `.like`, 
execution of the similarity and sorting computations of vectors is outsourced to 
a sidecar implementation which is managed by `superduperdb`.