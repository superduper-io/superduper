---
sidebar_position: 15
---

# Working with external data sources

Using the MongoDB query API, `superduperdb` supports data added from external data-sources.
The trick is to pass the `uri` parameter to an encoder, instead of the raw-data:

When doing this, `superduperdb` supports:

- web URLs
- URIs of objects in `s3` buckets

Here is an example where we add a `.pdf` file directly from a location 
on the public internet.

```python
import io
from PyPDF2 import PdfReader
from superduperdb.backends.mongodb import Collection

collection = Collection('pdf-files')


def load_pdf(bytes):
    text = []
    for page in PdfReader(io.BytesIO(bytes)).pages:
        text.append(page.extract_text())
    return '\n----NEW-PAGE----\n'.join(text)


pdf_enc = Encoder('my-pdf-encoder', decoder=load_pdf)

PDF_URI = (
    'https://papers.nips.cc/paper_files/paper/2012/file/'
    'c399862d3b9d6b76c8436e924a68c45b-Paper.pdf'
)

db.execute(
    collection.insert_one(Document({'txt': pdf_enc(uri=PDF_URI)}))
)
```

Now when the data is loaded from the database, it is loaded as text:

```python
>>> r = db.execute(collection.find_one())
>>> print(r['txt'])
```