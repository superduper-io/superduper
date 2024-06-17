
# Custom serialization

In this tutorial, we demonstrate how developers can flexibily and portably define
their own classes in `superduperdb`. These may be exported with `Component.export` 
and transported to other `superduperdb` deployments with `db.apply`.

To make our lives difficult, we'll include a data blob in the model, which should be serialized with the 
exported class:

```python
!curl -O https://superduperdb-public-demo.s3.amazonaws.com/text.json
import json

with open('text.json') as f:
    data = json.load(f)
```

<details>
<summary>Outputs</summary>

</details>

We'll define our own `Model` descendant, with a custom `.predict` method. 
We are free to define any of our own parameters to this class with a simple annotation in the header, since `Model` 
is a `dataclasses.dataclass`:

```python
from superduperdb import *


requires_packages(['openai', None, None])


class NewModel(Model):
    a: int = 2
    b: list

    def predict(self, x):
        return x * self.a
```

<details>
<summary>Outputs</summary>

</details>

If we want `b` to be saved as a blob in the `db.artifact_store` we can simply
annotate this in the `artifacts=...` parameter, supplying the serializer we would like to use:

```python
m = NewModel('test-hg', a=2, b=data, artifacts={'b': pickle_serializer})
```

<details>
<summary>Outputs</summary>

</details>

Now we can export the model:

```python
m.export('test-hg')
```

<details>
<summary>Outputs</summary>

</details>

```python
!cat test-hg/component.json
```

<details>
<summary>Outputs</summary>
<pre>
    \{
      "_base": "?test-hg",
      "_builds": \{
        "dill": \{
          "_path": "superduperdb.components.datatype.get_serializer",
          "method": "dill",
          "encodable": "artifact"
        \},
        "d0cd766789b72ffd8cb3d56484b02d8262dcc9b4": \{
          "_path": "superduperdb.components.datatype.Artifact",
          "datatype": "?dill",
          "blob": "&:blob:d0cd766789b72ffd8cb3d56484b02d8262dcc9b4"
        \},
        "pickle": \{
          "_path": "superduperdb.components.datatype.get_serializer",
          "method": "pickle",
          "encodable": "artifact"
        \},
        "e149b30249df8e7e2785fbbb58054cbe898a3cfd": \{
          "_path": "superduperdb.components.datatype.Artifact",
          "datatype": "?pickle",
          "blob": "&:blob:e149b30249df8e7e2785fbbb58054cbe898a3cfd"
        \},
        "test-hg": \{
          "_object": "?d0cd766789b72ffd8cb3d56484b02d8262dcc9b4",
          "b": "?e149b30249df8e7e2785fbbb58054cbe898a3cfd"
        \}
      \},
      "_files": \{\}
    \}
</pre>
</details>

```python
!ls test-hg/blobs/
```

<details>
<summary>Outputs</summary>
<pre>
    748ab67dbe58e1c269f83d72a77ad91cbc313c24
    d0cd766789b72ffd8cb3d56484b02d8262dcc9b4
    e149b30249df8e7e2785fbbb58054cbe898a3cfd

</pre>
</details>

The following cell works even after restarting the kernel.
That means the exported component is now completely portable!

```python
from superduperdb import *

c = Component.read('test-hg')

c.predict(2)
```

<details>
<summary>Outputs</summary>
<pre>
    4
</pre>
</details>
