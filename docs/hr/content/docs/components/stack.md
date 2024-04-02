# Stack

***Scope***

- Wraps multiple potentially interdependent components
- Allows developers to "apply" a range of functionality in a single declaration
- Allows decision makers and admins to manage "groups" of AI functionality

***Usage pattern***

```python
from superduperdb import Stack

stack = Stack(
    'my-stack',
    components=[
        table_1,
        model_1,
        model_2,
        listener_1,
        vector_index_1,
    ]
)

db.add(m)
```

***See also***

- [YAML stack syntax](../cluster_mode/yaml_syntax)