**`superduperdb.base.config`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/base/config.py)

## `BaseConfig` 

```python
BaseConfig(self) -> None
```
A base class for configuration dataclasses.

This class allows for easy updating of configuration dataclasses
with a dictionary of parameters.

## `CDCConfig` 

```python
CDCConfig(self,
     uri: Optional[str] = None,
     strategy: Union[superduperdb.base.config.PollingStrategy,
     superduperdb.base.config.LogBasedStrategy,
     NoneType] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| uri | The URI for the CDC service |
| strategy | The strategy to use for CDC |

Describes the configuration for change data capture.

## `CDCStrategy` 

```python
CDCStrategy(self,
     type: str) -> None
```
| Parameter | Description |
|-----------|-------------|
| type | The type of CDC strategy |

Base CDC strategy dataclass.

## `Cluster` 

```python
Cluster(self,
     compute: superduperdb.base.config.Compute = <factory>,
     vector_search: superduperdb.base.config.VectorSearch = <factory>,
     rest: superduperdb.base.config.Rest = <factory>,
     cdc: superduperdb.base.config.CDCConfig = <factory>) -> None
```
| Parameter | Description |
|-----------|-------------|
| compute | The URI for compute - None: run all jobs in local mode i.e. simple function call - "ray://host:port": Run all jobs on a remote ray cluster |
| vector_search | The URI for the vector search service - None: Run vector search on local - `f"http://{host}:{port}"`: Connect a remote vector search service |
| rest | The URI for the REST service - `f"http://{host}:{port}"`: Connect a remote vector search service |
| cdc | The URI for the change data capture service (if "None" then no cdc assumed) None: Run cdc on local as a thread. - `f"{http://{host}:{port}"`: Connect a remote cdc service |

Describes a connection to distributed work via Ray.

## `Compute` 

```python
Compute(self,
     uri: Optional[str] = None,
     compute_kwargs: Dict = <factory>) -> None
```
| Parameter | Description |
|-----------|-------------|
| uri | The URI for the compute service |
| compute_kwargs | The keyword arguments to pass to the compute service |

Describes the configuration for distributed computing.

## `Config` 

```python
Config(self,
     envs: dataclasses.InitVar[typing.Optional[typing.Dict[str,
     str]]] = None,
     data_backend: str = 'mongodb://localhost:27017/test_db',
     lance_home: str = '.superduperdb/vector_indices',
     artifact_store: Optional[str] = None,
     metadata_store: Optional[str] = None,
     cluster: superduperdb.base.config.Cluster = <factory>,
     retries: superduperdb.base.config.Retry = <factory>,
     downloads: superduperdb.base.config.Downloads = <factory>,
     fold_probability: float = 0.05,
     log_level: superduperdb.base.config.LogLevel = <LogLevel.INFO: 'INFO'>,
     logging_type: superduperdb.base.config.LogType = <LogType.SYSTEM: 'SYSTEM'>,
     bytes_encoding: superduperdb.base.config.BytesEncoding = <BytesEncoding.BYTES: 'Bytes'>,
     auto_schema: bool = True) -> None
```
| Parameter | Description |
|-----------|-------------|
| envs | The envs datas |
| data_backend | The URI for the data backend |
| lance_home | The home directory for the Lance vector indices, Default: .superduperdb/vector_indices |
| artifact_store | The URI for the artifact store |
| metadata_store | The URI for the metadata store |
| cluster | Settings distributed computing and change data capture |
| retries | Settings for retrying failed operations |
| downloads | Settings for downloading files |
| fold_probability | The probability of validation fold |
| log_level | The severity level of the logs |
| logging_type | The type of logging to use |
| bytes_encoding | The encoding of bytes in the data backend |
| auto_schema | Whether to automatically create the schema. If True, the schema will be created if it does not exist. |

The data class containing all configurable superduperdb values.

## `Downloads` 

```python
Downloads(self,
     folder: Optional[str] = None,
     n_workers: int = 0,
     headers: Dict = <factory>,
     timeout: Optional[int] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| folder | The folder to download files to |
| n_workers | The number of workers to use for downloading |
| headers | The headers to use for downloading |
| timeout | The timeout for downloading |

Describes the configuration for downloading files.

## `LogBasedStrategy` 

```python
LogBasedStrategy(self,
     type: str = 'logbased',
     resume_token: Optional[Dict[str,
     str]] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| resume_token | The resume token to use for log-based CDC |
| type | The type of CDC strategy |

Describes a log-based strategy for change data capture.

## `PollingStrategy` 

```python
PollingStrategy(self,
     type: 'str' = 'incremental',
     auto_increment_field: Optional[str] = None,
     frequency: float = 3600) -> None
```
| Parameter | Description |
|-----------|-------------|
| auto_increment_field | The field to use for auto-incrementing |
| frequency | The frequency to poll for changes |
| type | The type of CDC strategy |

Describes a polling strategy for change data capture.

## `Rest` 

```python
Rest(self,
     uri: Optional[str] = None,
     config: Optional[str] = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| uri | The URI for the REST service |
| config | The path to the config yaml file for the REST service |

Describes the configuration for the REST service.

## `Retry` 

```python
Retry(self,
     stop_after_attempt: int = 2,
     wait_max: float = 10.0,
     wait_min: float = 4.0,
     wait_multiplier: float = 1.0) -> None
```
| Parameter | Description |
|-----------|-------------|
| stop_after_attempt | The number of attempts to make |
| wait_max | The maximum time to wait between attempts |
| wait_min | The minimum time to wait between attempts |
| wait_multiplier | The multiplier for the wait time between attempts |

Describes how to retry using the `tenacity` library.

## `VectorSearch` 

```python
VectorSearch(self,
     uri: Optional[str] = None,
     type: str = 'in_memory',
     backfill_batch_size: int = 100) -> None
```
| Parameter | Description |
|-----------|-------------|
| uri | The URI for the vector search service |
| type | The type of vector search service |
| backfill_batch_size | The size of the backfill batch |

Describes the configuration for vector search.

