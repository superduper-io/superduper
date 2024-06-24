---
sidebar_label: Insert simple data
filename: insert_simple_data.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Insert simple data

After turning on auto_schema, we can directly insert data, and superduperdb will automatically analyze the data type, and match the construction of the table and datatype.

```python
from superduperdb import Document

table_or_collection = db['documents']

ids = db.execute(table_or_collection.insert([Document(data) for data in datas]))
select = table_or_collection.select()
```

<DownloadButton filename="insert_simple_data.md" />
