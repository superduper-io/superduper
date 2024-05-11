# Vector-search

:::note
Since vector-search is all-the-rage right now, 
here is the simplest possible iteration of semantic 
text-search with a `sentence_transformers` model, 
as an entrypoint to `superduperdb`.

Note that `superduperdb` is much-much more than vector-search
on text. Explore the docs to read about classical machine learning, 
computer vision, LLMs, fine-tuning and much much more!
:::


First let's get some data. This data are the markdown files 
of the very same documentation you are reading!
You can download other sample data-sets for testing `superduperdb`
by following [these lines of code](../reusable_snippets/get_useful_sample_data).


```python
import json
import requests 
r = requests.get('https://superduperdb-public-demo.s3.amazonaws.com/text.json')

with open('text.json', 'wb') as f:
    f.write(r.content)

with open('text.json', 'r') as f:
    data = json.load(f)        

print(data[0])
```

> output:
>
> ```
> # Anthropic
>    
> `superduperdb` allows users to work with `anthropic` API models.
>    
> Read more about this [here](/docs/docs/walkthrough/ai_models#anthropic).
> ```

Now we connect to SuperDuperDB, using MongoMock as a databackend.
Read more about connecting to SuperDuperDB [here](../core_api/connect) and
a semi-exhaustive list of supported data-backends for connecting [here](../reusable_snippets/connect_to_superduperdb).

```python
db = superduper('mongomock://test')

db['documents'].insert_many([{'txt': txt} for txt in data]).execute()
```

We are going to make these data searchable by activating a [`Model`](../apply_api/model) instance 
to compute vectors for each item inserted to the `"documents"` collection.
For that we'll use the [sentence-transformers](https://sbert.net/) integration to `superduperdb`.
Read more about the `[sentence_transformers` integration [here](../ai_integrations/sentence_transformers)
and [here](../../api/ext/sentence_transformers/).


```python
from superduperdb.ext.sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    identifier="test",
    datatype=datatype,
    predict_kwargs={"show_progress_bar": True},
    signature="*args,**kwargs",
    model="all-MiniLM-L6-v2",
    device="cpu",
    postprocess=lambda x: x.tolist(),
)
```

We can check that this model gives us what we want by evaluating an output 
on a single data-point. (Learn more about the various aspects of `Model` [here](../models/).)


```python
model.predict_one(data[0])
```

Now that we've verified that this model works, we can "activate" it for 
vector-search by creating a [`VectorIndex`](../apply_api/vector_index).


```python
vector_index = model.to_vector_index(select=db['documents'].find(), key='txt')
print(vector_index)
```

You will see that the `VectorIndex` contains a [`Listener`](../apply_api/listener) instance.
This instance wraps the model, and configures it to compute outputs 
on data inserted to the `"documents"` collection with the key `"txt"`.

To activate this index, we now do:

```python
db.apply(vector_index)
```

The `db.apply` command is a universal command for activating AI components in SuperDuperDB.

You will now see lots of output - the model-outputs/ vectors are computed 
and the various parts of the `VectorIndex` are registered in the system.

You can verify this with:

db.show()

To "use" the `VectorIndex` we can execute a vector-search query:


```python
query = db['documents'].like({'txt': 'Tell me about vector-search'}).find().limit(3)
cursor = query.execute()
```

This query will return a cursor of [`Document`](../fundamentals/document) instances.
To obtain the raw dictionaries, call the `.unpack()` command:


```python
for r in cursor:
    print(r.unpack())
```

You should see that the documents returned are relevant to the `like` part of the 
query.

Learn more about building queries with `superduperdb` [here](../execute_api).
