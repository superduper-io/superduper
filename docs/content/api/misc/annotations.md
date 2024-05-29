**`superduperdb.misc.annotations`** 

[Source code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/misc/annotations.py)

## `merge_docstrings` 

```python
merge_docstrings(cls)
```
| Parameter | Description |
|-----------|-------------|
| cls | Class to merge docstrings for. |

Decorator that merges Sphinx-styled class docstrings.

Decorator merges doc-strings from parent to child classes,
ensuring no duplicate parameters and correct indentation.

## `deprecated` 

```python
deprecated(f)
```
| Parameter | Description |
|-----------|-------------|
| f | function to deprecate |

Decorator to mark a function as deprecated.

This will result in a warning being emitted when the function is used.

## `component` 

```python
component(*schema: Dict)
```
| Parameter | Description |
|-----------|-------------|
| schema | schema for the component |

Decorator for creating a component.

## `requires_packages` 

```python
requires_packages(*packages,
     warn=False)
```
| Parameter | Description |
|-----------|-------------|
| packages | list of tuples of packages each tuple of the form (import_name, lower_bound/None, upper_bound/None, install_name/None) |
| warn | if True, warn instead of raising an exception |

Require the packages to be installed.

E.g. ('sklearn', '0.1.0', '0.2.0', 'scikit-learn')

## `extract_parameters` 

```python
extract_parameters(doc)
```
| Parameter | Description |
|-----------|-------------|
| doc | Sphinx-styled docstring. Docstring may have multiple lines |

Extracts and organizes parameter descriptions from a Sphinx-styled docstring.

## `replace_parameters` 

```python
replace_parameters(doc,
     placeholder: str = '!!!')
```
| Parameter | Description |
|-----------|-------------|
| doc | Sphinx-styled docstring. |
| placeholder | Placeholder to replace parameters with. |

Replace parameters in a doc-string with a placeholder.

## `SuperDuperDBDeprecationWarning` 

```python
SuperDuperDBDeprecationWarning(self,
     /,
     *args,
     **kwargs)
```
| Parameter | Description |
|-----------|-------------|
| args | *args of `DeprecationWarning` |
| kwargs | **kwargs of `DeprecationWarning` |

Specialized Deprecation Warning for fine grained filtering control.

