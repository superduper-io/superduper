# Component API

AI functionality in SuperDuperDB revolves around creating AI models, 
and configuring them to interact with data via the datalayer.

There are many decisions to be made and configured; for this SuperDuperDB
provides the `Component` abstraction.

The typical process is:

### 1. Create a component

Build your components, potentially including other subcomponents.

```python
from superduperdb import <ComponentClass>

component = <ComponentClass>(
    'identifier',
    **kwargs   # can include other components
)
```

### 2. Apply the component to the datalayer

"Applying" the component the `db` datalayer, also
applies all sub-components. So only 1 call is needed.

```python
db.apply(component)
```

### 3. Reload the component (if necessary)

The `.apply` command saves everything necessary to reload the component
in the SuperDuperDB system.

```python
reloaded = db.load('type_id', 'identifier')   # `type_id`
```