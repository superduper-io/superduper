**`superduperdb.base.document`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/base/document.py)

## `Document` 

```python
Document(self,
     *args,
     schema: Optional[ForwardRef('Schema')] = None,
     db: Optional[ForwardRef('Datalayer')] = None,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| args | *args for `dict` |
| schema | The schema to use. |
| db | The datalayer to use. |
| kwargs | **kwargs for `dict` |

A wrapper around an instance of dict or a Encodable.

The document data is used to dump that resource to
a mix of json-able content, ids and `bytes`

## `QueryUpdateDocument` 

```python
QueryUpdateDocument(self,
     *args,
     schema: Optional[ForwardRef('Schema')] = None,
     db: Optional[ForwardRef('Datalayer')] = None,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| args | *args for `dict` |
| schema | The schema to use. |
| db | The datalayer to use. |
| kwargs | **kwargs for `dict` |

A document that is used to update a document in a database.

This document is used to update a document in a database.
It is a subclass of Document.

