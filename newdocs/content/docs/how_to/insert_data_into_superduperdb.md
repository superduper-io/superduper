# Add images, audio or video from URIs

In this "how-to" guide, we demonstrate how to add images, audio or video to SuperDuperDB.

First, let's get a `Datalayer` instance, in order to demonstrate:


```python
import pymongo
from superduperdb import superduper
from superduperdb.db.mongodb.query import Collection

db = pymongo.MongoClient().documents

db = superduper(db)

collection = Collection('complexdata')
```

In order to encode, we create an `Encoder` instance:


```python
from superduperdb.container.encoder import Encoder
import io
import pickle
import PIL.Image

# encoder handles conversion to `bytes`
def encoder(image):
    buf = io.BytesIO()
    image.save(buf, format='png')
    return buf.getvalue()

# decoder handles conversion from `bytes` to Python
decoder = lambda x: PIL.Image.open(io.BytesIO(x))

enc = Encoder(identifier='my-enc', encoder=encoder, decoder=decoder)
```

We don't need to load our objects (images etc.) to add to the DB, we can use URIs instead:


```python
import glob
from superduperdb.container.document import Document as D

imgs = glob.glob('../img/*.png')

# wrap documents with `Document` in order so that SuperDuperDB knows how to handle
# wrap URI with `enc` to designate as "to-be-encoded"
# The URIs can be a mixture of `file://`, `http://`, `https://` and `s3://`
db.execute(
    collection.insert_many([
        D({'img': enc(uri=f'file://{img}')}) for img in imgs
    ], encoders=(enc,))
)
```


```python
r = db.execute(collection.find_one()).unpack()
r
```

We can verify that the image was properly stored:


```python
i = r['img'].convert('RGB')
i
```

We can also add Python objects directly:


```python
db.execute(collection.insert_one(D({'img': enc(i), 'direct': True})))
```

Verify:


```python
r = db.execute(collection.find_one({'direct': True})).unpack()
r['img']
```
