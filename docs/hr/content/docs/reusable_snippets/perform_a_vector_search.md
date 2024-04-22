---
sidebar_label: Perform a vector search
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Perform a vector search

```python
from superduperdb import Document

item = Document({indexing_key: sample_datapoint})
```

Once we have this search target, we can execute a search as follows:


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        select = query_table_or_collection.like(item, vector_index=vector_index_name, n=10).find()        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        select = query_table_or_collection.like(item)        
        ```
    </TabItem>
</Tabs>
```python
results = db.execute(select)
```

