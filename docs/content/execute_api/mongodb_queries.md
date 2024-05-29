# MongoDB select queries

SuperDuperDB supports the `pymongo` query API to build select queries.
There is one slight difference however, since queries built with `pymongo`'s formalism
are executed lazily:

Whereas in `pymongo` one might write:

```python
client.my_db.my_collection.find_one()
```

with `superduperdb` one would write:

```python
result = db['my_collection'].find_one().execute()
```
