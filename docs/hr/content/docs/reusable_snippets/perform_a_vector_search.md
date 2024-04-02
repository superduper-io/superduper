---
sidebar_label: Perform a vector search
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Perform a vector search

```python
from superduperdb import Document

item = Document({'x': datatype(sample_datapoint)})
```

Once we have this search target, we can execute a search as follows:


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        select = collection.find().like(sample_datapoint)        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        select = table.like(item)        
        ```
    </TabItem>
</Tabs>
```python
results = db.execute(select)
```

