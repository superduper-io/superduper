# SQL select queries

In order to support as many data-backends as possible, SuperDuperDB supports the `ibis` query API to build SQL queries. Note that in order to use these queries, developers should [first create tables and schemas](./data_encodings_and_schemas#table-schemas-in-sql).

With `superduperdb` one would write:

```python
t = db.load('table', 'my-table')
result = db.execute(t.filter(t.brand == 'Nike'))
```

Joins and other operations supported by `ibis` are also supported:

```python
s = db.load('table', 'my-table')
result = db.execute(t.filter(t.brand == 'Nike').join(t.id == s.my_table_id))
```

## Native SQL queries

Native SQL queries are also supported, provided that they do 
not require any special encoding of data (e.g. images, videos etc.):

```python
db.execute('SELECT * FROM my-table WHERE brand = Nike')
```