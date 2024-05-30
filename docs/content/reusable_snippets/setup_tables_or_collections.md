---
sidebar_label: Setup tables or collections
filename: setup_tables_or_collections.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Setup tables or collections

```python
from superduperdb.components.table import Table
from superduperdb import Schema

schema = Schema(identifier="schema", fields={"x": datatype})
table_or_collection = Table("documents", schema=schema)
db.apply(table_or_collection)
```

<DownloadButton filename="setup_tables_or_collections.md" />
