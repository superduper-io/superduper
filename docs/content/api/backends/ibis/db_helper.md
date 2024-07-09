**`superduper.backends.ibis.db_helper`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/backends/ibis/db_helper.py)

## `get_db_helper` 

```python
get_db_helper(dialect) -> superduper.backends.ibis.db_helper.DBHelper
```
| Parameter | Description |
|-----------|-------------|
| dialect | The dialect of the database. |

Get the insert processor for the given dialect.

## `ClickHouseHelper` 

```python
ClickHouseHelper(self,
     dialect)
```
| Parameter | Description |
|-----------|-------------|
| dialect | The dialect of the database. |

Helper class for ClickHouse database.

This class is used to convert byte data to base64 format for storage in the
database.

## `DBHelper` 

```python
DBHelper(self,
     dialect)
```
| Parameter | Description |
|-----------|-------------|
| dialect | The dialect of the database. |

Generic helper class for database.

