**`superduperdb.base.exceptions`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/base/exceptions.py)

## `DatabackendException` 

```python
DatabackendException(self,
     msg)
```
| Parameter | Description |
|-----------|-------------|
| msg | msg for BaseException |

DatabackendException.

## `BaseException` 

```python
BaseException(self,
     msg)
```
| Parameter | Description |
|-----------|-------------|
| msg | msg for Exception |

BaseException which logs a message after exception.

## `ComponentException` 

```python
ComponentException(self,
     msg)
```
| Parameter | Description |
|-----------|-------------|
| msg | msg for BaseException |

ComponentException.

## `ComponentInUseError` 

```python
ComponentInUseError(self,
     /,
     *args,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| args | *args for Exception |
| kwargs | **kwargs for Exception |

Exception raised when a component is already in use.

## `ComponentInUseWarning` 

```python
ComponentInUseWarning(self,
     /,
     *args,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| args | *args for Exception |
| kwargs | **kwargs for Exception |

Warning raised when a component is already in use.

## `MetadataException` 

```python
MetadataException(self,
     msg)
```
| Parameter | Description |
|-----------|-------------|
| msg | msg for BaseException |

MetadataException.

## `QueryException` 

```python
QueryException(self,
     msg)
```
| Parameter | Description |
|-----------|-------------|
| msg | msg for BaseException |

QueryException.

## `RequiredPackageVersionsNotFound` 

```python
RequiredPackageVersionsNotFound(self,
     /,
     *args,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| args | *args for ImportError |
| kwargs | **kwargs for ImportError |

Exception raised when one or more required packages are not found.

## `RequiredPackageVersionsWarning` 

```python
RequiredPackageVersionsWarning(self,
     /,
     *args,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| args | *args for ImportWarning |
| kwargs | **kwargs for ImportWarning |

Exception raised when one or more required packages are not found.

## `ServiceRequestException` 

```python
ServiceRequestException(self,
     msg)
```
| Parameter | Description |
|-----------|-------------|
| msg | msg for BaseException |

ServiceRequestException.

## `UnsupportedDatatype` 

```python
UnsupportedDatatype(self,
     msg)
```
| Parameter | Description |
|-----------|-------------|
| msg | msg for BaseException |

UnsupportedDatatype.

