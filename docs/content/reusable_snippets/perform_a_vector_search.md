---
sidebar_label: Perform a vector search
filename: perform_a_vector_search.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Perform a vector search

```python
from superduperdb import Document

def get_sample_item(key, sample_datapoint, datatype=None):
    if not isinstance(datatype, DataType):
        item = Document({key: sample_datapoint})
    else:
        item = Document({key: datatype(sample_datapoint)})

    return item

if compatible_key:
    item = get_sample_item(compatible_key, sample_datapoint, None)
else:
    item = get_sample_item(indexing_key, sample_datapoint, datatype=datatype)
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
        select = query_table_or_collection.like(item, vector_index=vector_index_name, n=10).limit(10)        
        ```
    </TabItem>
</Tabs>
```python
results = db.execute(select)
```

<DownloadButton filename="perform_a_vector_search.md" />
