**`superduper.components.component`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper.components/component.py)

## `ensure_initialized` 

```python
ensure_initialized(func)
```
| Parameter | Description |
|-----------|-------------|
| func | Decorator function. |

Decorator to ensure that the model is initialized before calling the function.

## `getdeepattr` 

```python
getdeepattr(obj,
     attr)
```
| Parameter | Description |
|-----------|-------------|
| obj | Object. |
| attr | Attribute. |

Get nested attribute with dot notation.

## `import_` 

```python
import_(r=None,
     path=None,
     db=None)
```
| Parameter | Description |
|-----------|-------------|
| r | Object to be imported. |
| path | Components directory. |
| db | Datalayer instance. |

Helper function for importing component JSONs, YAMLs, etc.

## `Component` 

```python
Component(self,
     identifier: str,
     db: dataclasses.InitVar[typing.Optional[ForwardRef('Datalayer')]] = None,
     uuid: str = None,
     *,
     artifacts: 'dc.InitVar[t.Optional[t.Dict]]' = None) -> None
```
| Parameter | Description |
|-----------|-------------|
| identifier | Identifier of the leaf. |
| db | Datalayer instance. |
| uuid | UUID of the leaf. |
| artifacts | A dictionary of artifacts paths and `DataType` objects |

Base class for all components in superduper.

Class to represent superduper serializable entities
that can be saved into a database.

