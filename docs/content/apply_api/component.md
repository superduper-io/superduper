# Apply API

:::info
In this section we re-use the datalayer variable `db` without further explanation.
Read more about how to build it [here](../core_api/connect) and what it is [here](../fundamentals/datalayer_overview).
:::

AI functionality in superduper revolves around creating AI models, 
and configuring them to interact with data via the datalayer.

There are many decisions to be made and configured; for this superduper
provides the `Component` abstraction.

The typical process is:

### 1. Create a component

Build your components, potentially including other subcomponents.

```python
from superduper import <ComponentClass>

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
in the superduper system.

```python
reloaded = db.load('type_id', 'identifier')   # `type_id`
```

### 4. Export the component (to share/ migrate)

The `.export` command saves the entirety of the component's parameters, 
inline code and artifacts in a directory:

```python
component.export('my_component')
```

The directory structure looks like this.
It contains the meta-data of the component as
well as a "mini-artifact-store". Together
these items make the export completely portable.

```
my_component
|_component.json // meta-data and imports of component
|_blobs   // directory of blobs used in the component
| |_234374364783643764
| |_574759874238947320
|_files   // directory of files used in the component
  |_182372983721897389
  |_982378978978978798
```

You can read about the serialization mechanism [here](../production/superduper_protocol.md.md).

## Read more

```mdx-code-block
import DocCardList from '@theme/DocCardList';

<DocCardList />
```
