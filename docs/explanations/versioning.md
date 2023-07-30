# Component versioning

SuperDuperDB has a component versioning system. This versioning system applies to subclasses of 
`Component`:

- [models](modelz)
- [encoders](encoders)
- [watchers](watchers)
- [vector-indexes](vectorsearch)

Here's how it works:

When users create a component, they are required to choose an `identifier`.
The `identifier` must be unique for each of the distinct classes above.

After the `identifier` has been chosen, SuperDuperDB creates a version
starting at 0, and incremented each time an item with `identifier` is recreated. 

For example the following would create two versions of a `Model`, identified by `"test-model"`:

```python
>>> db.add(Model(identifier='test-model', object=my_model))
>>> db.add(Model(identifier='test-model', object=my_model))
>>> db.show('model')   # shows distinct identifiers
['test-model']
>>> db.show('model', 'test-model')       # shows distinct versions
[0, 1]
>>> db.show('model', 'test-model', 0)     # shows full meta-data record of model version
```

When creating a `Component`, other `Component` contained in the first are created inline.

When removing a `Component` the versioning system is interrogated and if they play a role in other `Component` instances, they may not be deleted. If deletion is forced, they are marked as hidden, rather than deleted.

For example the following raises an `Exception`:

```
>>> enc = Encoder('my-enc', encode=lambda x: x, decode=lambda x: x)
>>> db.add(Model(identifier='test-model', object=my_model, enc=enc))
>>> db.remove('encoder', 'my-enc')
```