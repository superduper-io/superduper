**`superduperdb.misc.special_dicts`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/misc/special_dicts.py)

## `diff` 

```python
diff(r1,
     r2)
```
| Parameter | Description |
|-----------|-------------|
| r1 | Dict |
| r2 | Dict |

Get the difference between two dictionaries.

```python
_diff({'a': 1, 'b': 2}, {'a': 2, 'b': 2})
# {'a': (1, 2)}
_diff({'a': {'c': 3}, 'b': 2}, {'a': 2, 'b': 2})
# {'a': ({'c': 3}, 2)}
```

## `SuperDuperFlatEncode` 

```python
SuperDuperFlatEncode(self,
     /,
     *args,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| args | *args for `dict` |
| kwargs | **kwargs for `dict` |

Dictionary for representing flattened encoding data.

## `MongoStyleDict` 

```python
MongoStyleDict(self,
     /,
     *args,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| args | *args for `dict` |
| kwargs | **kwargs for `dict` |

Dictionary object mirroring how MongoDB handles fields.

## `IndexableDict` 

```python
IndexableDict(self,
     ordered_dict)
```
| Parameter | Description |
|-----------|-------------|
| ordered_dict | OrderedDict |

IndexableDict.

```python
# Example:
# -------
d = IndexableDict({'a': 1, 'b': 2})
d[0]
# 1
```

```python
d[1]
# 2
```

