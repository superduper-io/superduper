# `DataInit`

- Used to automatically insert initialization data during application build.

***Usage pattern***

```python
from superduper.components.dataset import DataInit
data = [{"x": i, "y": [1, 2, 3]} for i in range(10)]
data_init = DataInit(data=data, table="documents", identifier="test_data_init")

db.apply(data_init)
```

***Explanation***

- When db.apply(data_init) is executed, DataInit inserts data into the specified table.