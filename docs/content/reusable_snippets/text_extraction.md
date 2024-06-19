```python
# <tab: PDF>
from PyPDF2 import PdfReader

from superduperdb import model


@model(flatten=True, model_update_kwargs={'document_embedded': False})
def text_extraction(file_path):
    reader = PdfReader(file_path)
    
    texts = []
    for i, page in tqdm(enumerate(reader.pages)):
        text = page.extract_text() 
        texts.append(text)
    return texts
```


```python

```
