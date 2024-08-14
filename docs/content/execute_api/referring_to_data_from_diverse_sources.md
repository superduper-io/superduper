---
sidebar_position: 15
---

# Working with external data sources

Superduper supports data added from external data-sources.
When doing this, Superduper supports:

- web URLs
- URIs of objects in `s3` buckets

The trick is to pass the `uri` parameter to an encoder, instead of the raw-data.
Here is an example where we add a `.pdf` file directly from a location 
on the public internet.

```python
import io
from PyPDF2 import PdfReader

def load_pdf(bytes):
    text = []
    for page in PdfReader(io.BytesIO(bytes)).pages:
        text.append(page.extract_text())
    return '\n----NEW-PAGE----\n'.join(text)

# no `encoder=...` parameter required since text is not converted to `.pdf` format
pdf_enc = Encoder('my-pdf-encoder', decoder=load_pdf)

PDF_URI = (
    'https://papers.nips.cc/paper_files/paper/2012/file/'
    'c399862d3b9d6b76c8436e924a68c45b-Paper.pdf'
)

# This command inserts a record which refers to this URI
# and also downloads the content from the URI and saves
# it in the record
db['pdf-files'].insert_one(Document({'txt': pdf_enc(uri=PDF_URI)})).execute()
```

Now when the data is loaded from the database, it is loaded as text:

```python
>>> r = collection.find_one().execute()
>>> print(r['txt'])
```