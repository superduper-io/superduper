**`superduperdb.backends.mongodb.artifacts`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/backends/mongodb/artifacts.py)

## `upload_folder` 

```python
upload_folder(path,
     file_id,
     fs,
     parent_path='')
```
| Parameter | Description |
|-----------|-------------|
| path | The path to the folder to upload |
| file_id | The file_id of the folder |
| fs | The GridFS object |
| parent_path | The parent path of the folder |

Upload folder to GridFS.

## `MongoArtifactStore` 

```python
MongoArtifactStore(self,
     conn,
     name: str)
```
| Parameter | Description |
|-----------|-------------|
| conn | MongoDB client connection |
| name | Name of database to host filesystem |

Artifact store for MongoDB.

