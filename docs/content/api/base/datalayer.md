**`superduper.base.datalayer`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/base/datalayer.py)

## `Datalayer` 

```python
Datalayer(self,
     databackend: superduper.backends.base.data_backend.BaseDataBackend,
     metadata: superduper.backends.base.metadata.MetaDataStore,
     artifact_store: superduper.backends.base.artifacts.ArtifactStore,
     compute: superduper.backends.base.compute.ComputeBackend = <superduper.backends.local.compute.LocalComputeBackend object at 0x291ee3510>)
```
| Parameter | Description |
|-----------|-------------|
| databackend | Object containing connection to Datastore. |
| metadata | Object containing connection to Metadatastore. |
| artifact_store | Object containing connection to Artifactstore. |
| compute | Object containing connection to ComputeBackend. |

Base database connector for superduper.

## `LoadDict` 

```python
LoadDict(self,
     database: superduper.base.datalayer.Datalayer,
     field: Optional[str] = None,
     callable: Optional[Callable] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| database | Instance of Datalayer. |
| field | (optional) Component type identifier. |
| callable | (optional) Callable function on key. |

Helper class to load component identifiers with on-demand loading from the database.

