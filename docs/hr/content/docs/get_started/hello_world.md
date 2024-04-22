---
sidebar_position: 5
tags:
  - quickstart
---

# Your first `superduperdb` program

:::note
Since vector-search is all-the-rage right now, 
here is the simplest possible iteration of semantic 
text-search with a `sentence_transformers` model, 
as an entrypoint to `superduperdb`.

Note that `superduperdb` is much-much more than vector-search
on text. Explore the docs to read about classical machine learning, 
computer vision, LLMs, fine-tuning and much much more!
:::

To check that everything is working correctly cut and paste this code into a Jupyter notebook,
script or IPython session.

```python
import json
import requests 

from superduperdb import Stack, Document, VectorIndex, Listener, vector, superduper
from superduperdb.ext.sentence_transformers.model import SentenceTransformer
from superduperdb.backends.mongodb import Collection

r = requests.get('https://superduperdb-public-demo.s3.amazonaws.com/text.json')

with open('text.json', 'wb') as f:
    f.write(r.content)

with open('text.json', 'r') as f:
    data = json.load(f)        

db = superduper('mongomock://test')

db.execute(
    Collection('documents').insert_many([Document({'txt': r}) for r in data])
)

datatype = vector(shape=384, identifier="my-vec")

model = SentenceTransformer(
    identifier="test",
    datatype=datatype,
    predict_kwargs={"show_progress_bar": True},
    signature="*args,**kwargs",
    model="all-MiniLM-L6-v2",
    device="cpu",
    postprocess=lambda x: x.tolist(),
)

listener = Listener(
    identifier="my-listener",
    key="txt",
    model=model,
    select=Collection('documents').find(),
    active=True,
    predict_kwargs={}
)

vector_index = VectorIndex(
    identifier="my-index",
    indexing_listener=listener,
    measure="cosine"
)

db.apply(vector_index)

print(db.execute(
    Collection('documents')
        .like({'txt': 'Tell me about vector-indexes'}, vector_index='my-index')
        .find_one()
))
```

:::warning
If this doesn't work then something is wrong 🙉 - please report [an issue on GitHub](https://github.com/SuperDuperDB/superduperdb/issues).
:::

:::tip
This example deploys a vector-index with a model we have chosen and self-hosted,
on some text-data (`superduperdb` documentation). The single execution
`db.apply` sets everything up, and the `db.execute` call, executes a vector-search 
query-by-meaning. All of the parts `model`, `listener`, `data`, `vector_index`
are fully configurable and adaptable to your application's needs.
Read more in the main documentation!
:::