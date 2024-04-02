---
sidebar_position: 16
---

# Working with and inserting large pieces of data

:::warning
This functionality is currently only supported by the MongDB API
:::

Some applications require large data-blobs and objects, which are larger than the objects which are supported by the underlying database.

For example:

- MongoDB supports documents up to `16MB`
- SQLite `BINARY` has a default limit of `1GB`

In addition, for some applications, which are very read-heavy (for example training CNN on a database of images), storing data directly in the database, can lead to impaired database performance.

In such cases, `superduperdb` supports hybrid storage, where large data blobs, are stored on the local filesystem.

`superduperdb` supports this hybrid storage, via `env` variable or configuration.

The `downloads_folder` configuration is by default `None`, telling the system to save all data blobs directly 
in tables/ collectios. To enable the system to save downloads separately, set:

```python
CFG.downloads_folder = '<path-to-downloads-folder>'
```
... or

```bash
export SUPERDUPERDB_DOWNLOADS_FOLDER='<path-to-downloads-folder>'
```

Once this has been configured, inserting data proceeds exactly as before, with the difference 
that items inserted via URI, are saved on the local filesystem, and referred to in the database.

That is, when a command such as the one below, is executed, a reference to the data is stored in the database
instead of the raw-data itself. The raw-data is stored in the `CFG.downloads_folder` directory:

```python
uris = ... # list of URIs of content to be downloaded (local files, web URLs, s3 URIs)
db.execute(
    collection.insert_many([Document({'content': encoder(uri=uri)}) for uri in uris])
)
# Jobs are now queued which download the content behind the URIs and save it in `CFG.downloads_folder`
```

Read [here](./referring_to_data_from_diverse_sources.md) for more details on using URIs to insert data.