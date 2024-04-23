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

```python
from superduperdb.components.model import QueryModel

vector_search_model = QueryModel(
    identifier="VectorSearch",
    select=select,
    postprocess=lambda docs: [{"text": doc[indexing_key], "_source": doc["_source"]} for doc in docs]
)
vector_search_model.db = db
```

```python
vector_search_model.predict_one(query=query)
```

