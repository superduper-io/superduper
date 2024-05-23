# Select outputs of upstream listener

:::note
This is useful if you have performed a first step, such as pre-computing 
features, or chunking your data. You can use this query to 
operate on those outputs.
:::


```python
indexing_key = upstream_listener.outputs_key
select = upstream_listener.outputs_select
```
