---
sidebar_label: Perform a vector search
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Perform a vector search

- `item` is the item which is to be encoded
- `dt` is the `DataType` instance to apply

```python
from superduperdb import Document

item = Document({'my_key': dt(item)})
```

Once we have this search target, we can execute a search as follows:


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb.backends.mongodb import Collection
        
        collection = Collection('documents')
        
        select = collection.find().like(item)        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        
        # Table was created earlier, before preparing vector-search
        table = db.load('table', 'documents')
        
        select = table.like(item)        
        ```
    </TabItem>
</Tabs>
```python
results = db.execute(select)
```

