---
sidebar_label: Create Vector Search Model
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Create Vector Search Model

```python
from superduperdb.base.variables import Variable
item = {indexing_key: Variable('query')}
```

```python
from superduperdb.components.model import QueryModel

vector_search_model = QueryModel(
    identifier="VectorSearch",
    select=query_table_or_collection.like(item, vector_index=vector_index_name, n=5).select(),
    # The _source is the identifier of the upstream data, which can be used to locate the data from upstream sources using `_source`.
    postprocess=lambda docs: [{"text": doc[indexing_key], "_source": doc["_source"]} for doc in docs],
    db=db
)
```

```python
vector_search_model.predict(query=query)
```

