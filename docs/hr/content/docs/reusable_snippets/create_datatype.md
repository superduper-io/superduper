---
sidebar_label: Create datatype
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Create datatype

```python
# <testing:>
from superduperdb import superduper

db = superduper("mongomock://test")
```


<Tabs>
    <TabItem value="Vector" label="Vector" default>
        ```python
        from superduperdb import vector
        
        datatype = vector(shape=(3, ))
        origin_data = [0.1, 0.2, 0.3]        
        ```
    </TabItem>
    <TabItem value="Tensor" label="Tensor" default>
        ```python
        from superduperdb.ext.torch import tensor
        import torch
        
        datatype = tensor(torch.float, shape=(32, ))
        origin_data = torch.randn(32)        
        ```
    </TabItem>
    <TabItem value="Array" label="Array" default>
        ```python
        from superduperdb.ext.numpy import array
        import numpy as np
        
        datatype = array(dtype="float64", shape=(32, ))
        origin_data = np.random.randn(32)        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        from superduperdb.ext.pillow import pil_image
        import PIL.Image
        
        !wget https://raw.githubusercontent.com/openai/CLIP/main/CLIP.png
        
        datatype = pil_image
        origin_data = PIL.Image.open("CLIP.png")        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        !wget https://ccrma.stanford.edu/workshops/mir2014/audio/simpleLoop.wav
        
        import librosa
        from superduperdb.ext.numpy import array
        import numpy as np
        from IPython.display import Audio
        
        datatype = array(dtype="float32", shape=(None, ))
        origin_data, fs = librosa.load('simpleLoop.wav')
        
        Audio(origin_data, rate=fs)        
        ```
    </TabItem>
    <TabItem value="Video" label="Video" default>
        ```python
        from superduperdb import DataType
        
        # Create an instance of the Encoder with the identifier 'video_on_file' and load_hybrid set to False
        datatype = DataType(
            identifier='video_on_file',
            encodable='artifact',
        )        
        ```
    </TabItem>
    <TabItem value="Custom-in-DB" label="Custom-in-DB" default>
        ```python
        ...        
        ```
    </TabItem>
    <TabItem value="Custom-Artifact" label="Custom-Artifact" default>
        ```python
        ...        
        ```
    </TabItem>
</Tabs>
## Interact with the database.

```python
db.add(datatype)
```

### Insert data into the database

```python
from superduperdb.backends.mongodb import Collection
from superduperdb import Document
collection = Collection("data")
```

```python
db.execute(collection.insert_one(Document({"x": datatype(origin_data)})))
```

### Read data from the database

```python
data = db.execute(collection.find_one())
data
```

**Restore data to its original form**

```python
data.unpack()
```

```python
data.unpack()["x"]
```

