---
sidebar_label: Multimodal vector search
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Multimodal vector search

<!-- TABS -->
## Start your system


<Tabs>
    <TabItem value="Development" label="Development" default>
        ```python
        # Nothing to do here (everything runs in-process)        
        ```
    </TabItem>
    <TabItem value="Experimental Cluster" label="Experimental Cluster" default>
        ```python
        !python -m superduperdb local_cluster        
        ```
    </TabItem>
    <TabItem value="Docker-Compose" label="Docker-Compose" default>
        ```python
        !make testenv_image
        !make testenv_init        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Connect to SuperDuperDB


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb import superduper
        
        db = superduper('mongodb://localhost:27017/documents')        
        ```
    </TabItem>
    <TabItem value="SQLite" label="SQLite" default>
        ```python
        from superduperdb import superduper
        
        db = superduper('sqlite://my_db.db')        
        ```
    </TabItem>
    <TabItem value="MySQL" label="MySQL" default>
        ```python
        from superduperdb import superduper
        
        user = 'superduper'
        password = 'superduper'
        port = 3306
        host = 'localhost'
        database = 'test_db'
        
        db = superduper(f"mysql://{user}:{password}@{host}:{port}/{database}")        
        ```
    </TabItem>
    <TabItem value="Oracle" label="Oracle" default>
        ```python
        from superduperdb import superduper
        
        user = 'sa'
        password = 'Superduper#1'
        port = 1433
        host = 'localhost'
        
        db = superduper(f"mssql://{user}:{password}@{host}:{port}")        
        ```
    </TabItem>
    <TabItem value="PostgreSQL" label="PostgreSQL" default>
        ```python
        from superduperdb import superduper
        
        user = 'superduper'
        password = 'superduper'
        port = 5432
        host = 'localhost'
        database = 'test_db'
        
        db = superduper(f"postgres://{user}:{password}@{host}:{port}/{database}")        
        ```
    </TabItem>
    <TabItem value="Snowflake" label="Snowflake" default>
        ```python
        from superduperdb import superduper
        
        user = "superduperuser"
        password = "superduperpassword"
        account = "XXXX-XXXX"  # ORGANIZATIONID-USERID
        database = "FREE_COMPANY_DATASET/PUBLIC"
        
        snowflake_uri = f"snowflake://{user}:{password}@{account}/{database}"
        
        db = superduper(
            snowflake_uri, 
            metadata_store='sqlite:///your_database_name.db',
        )        
        ```
    </TabItem>
    <TabItem value="Clickhouse" label="Clickhouse" default>
        ```python
        from superduperdb import superduper
        
        user = 'default'
        password = ''
        port = 8123
        host = 'localhost'
        
        db = superduper(f"clickhouse://{user}:{password}@{host}:{port}", metadata_store=f'mongomock://meta')        
        ```
    </TabItem>
    <TabItem value="DuckDB" label="DuckDB" default>
        ```python
        from superduperdb import superduper
        
        db = superduper('duckdb://mydb.duckdb')        
        ```
    </TabItem>
    <TabItem value="Pandas" label="Pandas" default>
        ```python
        from superduperdb import superduper
        
        db = superduper(['my.csv'], metadata_store=f'mongomock://meta')        
        ```
    </TabItem>
    <TabItem value="MongoMock" label="MongoMock" default>
        ```python
        from superduperdb import superduper
        
        db = superduper('mongomock:///test_db')        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Get useful sample data


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        !curl -O https://superduperdb-public-demo.s3.amazonaws.com/text.json
        import json
        
        with open('text.json', 'r') as f:
            data = json.load(f)        
        ```
    </TabItem>
    <TabItem value="PDF" label="PDF" default>
        ```python
        !curl -O https://superduperdb-public-demo.s3.amazonaws.com/pdfs.zip && unzip pdfs.zip
        import os
        
        data = [f'pdfs/{x}' for x in os.listdir('./pdfs')]
        data        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        !curl -O s3://superduperdb-public-demo/images.zip && unzip images.zip
        import os
        
        data = [f'images/{x}' for x in os.listdir('./images')]        
        ```
    </TabItem>
    <TabItem value="Video" label="Video" default>
        ```python
        !curl -O s3://superduperdb-public-demo/videos.zip && unzip videos.zip
        import os
        
        data = [f'videos/{x}' for x in os.listdir('./videos')]        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        !curl -O s3://superduperdb-public-demo/audio.zip && unzip audio.zip
        import os
        
        data = [f'audios/{x}' for x in os.listdir('./audios')]        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Create datatype

Data types such as "text" or "integer" which are natively support by your `db.databackend` don't need a datatype.

```python
datatype = None
```

Otherwise do one of the following:


<Tabs>
    <TabItem value="PDF" label="PDF" default>
        ```python
        !pip install PyPDF2
        from superduperdb import DataType
        from superduperdb.components.datatype import File
        
        datatype = DataType('pdf', encodable='file')        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        from superduperdb.ext.pillow import pil_image
        import PIL.Image
        
        datatype = pil_image        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        from superduperdb.ext.numpy import array
        from superduperdb import DataType
        import scipy.io.wavfile
        import io
        
        
        def encoder(data):
            buffer = io.BytesIO()
            fs = data[0]
            content = data[1]
            scipy.io.wavfile.write(buffer, fs, content)
            return buffer.getvalue()
        
        
        def decoder(data):
            buffer = io.BytesIO(data)
            content = scipy.io.wavfile.read(buffer)
            return content
        
        
        datatype = DataType(
            'wav',
            encoder=encoder,
            decoder=decoder,
            encodable='artifact',
        )        
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
</Tabs>
<!-- TABS -->
## Setup tables or collections


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        # Note this is an optional step for MongoDB
        # Users can also work directly with `DataType` if they want to add
        # custom data
        from superduperdb import Schema
        
        schema = None
        if isinstance(datatype, DataType):
            schema = Schema(fields={'x': datatype})
            db.add(schema)        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb.backends.ibis import Table
        from superduperdb.backends.ibis.field_types import FieldType
        
        if isinstance(datatype, DataType):
            schema = Schema(fields={'x': datatype})
        else:
            schema = Schema(fields={'x': FieldType(datatype)})
        db.add(Table('documents', schema=schema))        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Insert data

In order to create data, we need to create a `Schema` for encoding our special `Datatype` column(s) in the databackend.

```python
N_DATA = round(len(data) - len(data) // 4)
```


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb import Document
        
        if schema is None:
            data = Document([{'x': datatype(x)} for x in data])    
            db.execute(collection.insert_many(data[:N_DATA]))
        else:
            data = Document([{'x': x} for x in data])    
            db.execute(collection.insert_many(data[:N_DATA], schema='my_schema'))        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb import Document
        
        db.execute(table.insert([Document({'x': x}) for x in data[:N_DATA]]))        
        ```
    </TabItem>
</Tabs>
```python
sample_datapoint = data[-1]
```

<!-- TABS -->
## Apply a chunker for search


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        from superduperdb import objectmodel
        
        CHUNK_SIZE = 200
        
        @objectmodel(flatten=True, model_update_kwargs={'document_embedded': False})
        def chunker(text):
            text = text.split()
            chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
            return chunks        
        ```
    </TabItem>
    <TabItem value="PDF" label="PDF" default>
        ```python
        !pip install PyPDF2
        from superduperdb import objectmodel
        
        CHUNK_SIZE = 500
        
        @objectmodel(flatten=True, model_update_kwargs={'document_embedded': False})
        def chunker(pdf_file):
            reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(reader.pages)
            print(f'Number of pages {num_pages}')
            text = []    
            for i in range(num_pages):
                page = reader.pages[i]        
                page_text = page.extract_text()
                text.append(page_text)
            text = '\n\n'.join(text)
            chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
            return chunks        
        ```
    </TabItem>
    <TabItem value="Video" label="Video" default>
        ```python
        !pip install opencv-python
        import cv2
        import tqdm
        from PIL import Image
        from superduperdb.ext.pillow import pil_image
        from superduperdb import ObjectModel, Schema
        
        
        @objectmodel(
            flatten=True,
            model_update_kwargs={'document_embedded': False},
            output_schema=Schema(identifier='output-schema', fields={'image': pil_image}),
        )
        def chunker(video_file):
            # Set the sampling frequency for frames
            sample_freq = 10
            
            # Open the video file using OpenCV
            cap = cv2.VideoCapture(video_file)
            
            # Initialize variables
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            extracted_frames = []
            progress = tqdm.tqdm()
        
            # Iterate through video frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get the current timestamp based on frame count and FPS
                current_timestamp = frame_count // fps
                
                # Sample frames based on the specified frequency
                if frame_count % sample_freq == 0:
                    extracted_frames.append({
                        'image': Image.fromarray(frame[:,:,::-1]),  # Convert BGR to RGB
                        'current_timestamp': current_timestamp,
                    })
                frame_count += 1
                progress.update(1)
            
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            
            # Return the list of extracted frames
            return extracted_frames        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        from superduperdb import objectmodel, Schema
        
        CHUNK_SIZE = 10  # in seconds
        
        @objectmodel(
            flatten=True,
            model_update_kwargs={'document_embedded': False},
            output_schema=Schema(identifier='output-schema', fields={'audio': datatype}),
        )
        def chunker(audio):
            chunks = []
            for i in range(0, len(audio), CHUNK_SIZE):
                chunks.append(audio[1][i: i + CHUNK_SIZE])
            return [(audio[0], chunk) for chunk in chunks]        
        ```
    </TabItem>
</Tabs>
Now we apply this chunker to the data by wrapping the chunker in `Listener`:

```python
db.add(
    Listener(
        model=chunker,
        select=select,
        key='x',
    )
)
```

<!-- TABS -->
## Build multimodal embedding models

Some embedding models such as [CLIP](https://github.com/openai/CLIP) come in pairs of `model` and `compatible_model`.
Otherwise:

```python
compatible_model = None
```


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        from superduperdb.ext.sentence_transformers import SentenceTransformer
        
        # Load the pre-trained sentence transformer model
        model = SentenceTransformer(identifier='all-MiniLM-L6-v2')        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        import torch
        import clip
        from torchvision import transforms
        from superduperdb.ext.torch import TorchModel
        
        class CLIPVisionEmbedding:
            def __init__(self):
                # Load the CLIP model
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model, self.preprocess = clip.load("RN50", device=self.device)
                
            def preprocess(self, image):
                # Load and preprocess the image
                image = self.preprocess(image).unsqueeze(0).to(self.device)
                return image
                
        model = CLIPVisionEmbedding()
        model = TorchModel(identifier='clip-vision', object=model.model, preprocess=model.preprocess, forward_method='encode_image')        
        ```
    </TabItem>
    <TabItem value="Text+Image" label="Text+Image" default>
        ```python
        
        import torch
        import clip
        from torchvision import transforms
        from superduperdb import Model
        from superduperdb.ext.torch import TorchModel
        
        class CLIPTextEmbedding:
            def __init__(self):
                # Load the CLIP model
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model, _ = clip.load("RN50", device=self.device)
                
            def __call__(self, text):
                features = clip.tokenize([text])
                return self.model.encode_text(features)
                
        model = CLIPTextEmbedding()
        superdupermodel_text = Model(identifier='clip-text', object=model)
        
        class CLIPVisionEmbedding:
            def __init__(self):
                # Load the CLIP model
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model, self.preprocess = clip.load("RN50", device=self.device)
                
            def preprocess(self, image):
                # Load and preprocess the image
                image = self.preprocess(image).unsqueeze(0).to(self.device)
                return image
                
        model = TorchModel(identifier='clip-vision', object=model.model, preprocess=model.preprocess, forward_method='encode_image')
        compatible_model = CLIPVisionEmbedding()        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        !pip install librosa
        import librosa
        import numpy as np
        from superduperdb import Model
        
        def audio_embedding(audio_file):
            # Load the audio file
            y, sr = librosa.load(audio_file)
            mfccs = librosa.feature.mfcc(y=y, sr=sr)
            return mfccs
        
        model= Model(identifier='my-model-audio', object=audio_embedding)        
        ```
    </TabItem>
</Tabs>
## Create vector-index


<Tabs>
    <TabItem value="1-Modality" label="1-Modality" default>
        ```python
        from superduperdb import VectorIndex, Listener
        
        jobs, _ db.add(
            VectorIndex(
                'my-vector-index',
                indexing_listener=Listener(
                    key='<my_key>',      # the `Document` key `model` should ingest to create embedding
                    select=select,       # a `Select` query telling which data to search over
                    model=model,         # a `_Predictor` how to convert data to embeddings
                )
            )
        )        
        ```
    </TabItem>
    <TabItem value="2-Modalities" label="2-Modalities" default>
        ```python
        from superduperdb import VectorIndex, Listener
        
        jobs, _ db.add(
            VectorIndex(
                'my-vector-index',
                indexing_listener=Listener(
                    key='<my_key>',      # the `Document` key `model` should ingest to create embedding
                    select=select,       # a `Select` query telling which data to search over
                    model=model,         # a `_Predictor` how to convert data to embeddings
                ),
                compatible_listener=Listener(
                    key='<other_key>',      # the `Document` key `model` should ingest to create embedding
                    model=compatible_model,         # a `_Predictor` how to convert data to embeddings
                    active=False,
                )
            )
        )        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Perform a vector search

```python
from superduperdb import Document

item = Document({'x': datatype(sample_datapoint)})
```

Once we have this search target, we can execute a search as follows:


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        select = collection.find().like(sample_datapoint)        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        select = table.like(item)        
        ```
    </TabItem>
</Tabs>
```python
results = db.execute(select)
```

