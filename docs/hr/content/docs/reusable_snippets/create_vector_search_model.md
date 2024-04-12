---
sidebar_label: Create Vector Search Model
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Create Vector Search Model

```python
from superduperdb.base.serializable import Variable
item = {indexing_key: Variable('query')}
```


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb.components.model import QueryModel
        
        vector_search_model = QueryModel(
            identifier="VectorSearch",
            select=query_table_or_collection.like(item, vector_index=vector_index_name, n=10).find(),
            postprocess=lambda docs: [{"_id": doc["_id"], "text": doc[indexing_key], "_source": doc["_source"]} for doc in docs]
        )
        vector_search_model.db = db        
        ```
    </TabItem>
</Tabs>
