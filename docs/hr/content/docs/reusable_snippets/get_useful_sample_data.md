---
sidebar_label: Get useful sample data
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Get useful sample data

```python
from superduperdb import dtype

```


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        !curl -O https://superduperdb-public-demo.s3.amazonaws.com/text.json
        import json
        
        with open('text.json', 'r') as f:
            data = json.load(f)
        sample_datapoint = "What is mongodb?"
        
        chunked_model_datatype = dtype('str')        
        ```
    </TabItem>
    <TabItem value="PDF" label="PDF" default>
        ```python
        !curl -O https://superduperdb-public-demo.s3.amazonaws.com/pdfs.zip && unzip -o pdfs.zip
        import os
        
        data = [f'pdfs/{x}' for x in os.listdir('./pdfs')]
        
        sample_datapoint = data[-1]
        chunked_model_datatype = dtype('str')        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        !curl -O s3://superduperdb-public-demo/images.zip && unzip images.zip
        import os
        from PIL import Image
        
        data = [f'images/{x}' for x in os.listdir('./images')]
        data = [ Image.open(path) for path in data]
        sample_datapoint = data[-1]
        
        from superduperdb.ext.pillow import pil_image
        chunked_model_datatype = pil_image        
        ```
    </TabItem>
    <TabItem value="Video" label="Video" default>
        ```python
        !curl -O s3://superduperdb-public-demo/videos.zip && unzip videos.zip
        import os
        
        data = [f'videos/{x}' for x in os.listdir('./videos')]
        sample_datapoint = data[-1]
        
        from superduperdb.ext.pillow import pil_image
        chunked_model_datatype = pil_image        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        !curl -O s3://superduperdb-public-demo/audio.zip && unzip audio.zip
        import os
        
        data = [f'audios/{x}' for x in os.listdir('./audios')]
        sample_datapoint = data[-1]
        chunked_model_datatype = dtype('str')        
        ```
    </TabItem>
</Tabs>
