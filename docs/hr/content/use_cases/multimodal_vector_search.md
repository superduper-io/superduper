---
sidebar_label: Multimodal vector search
filename: multimodal_vector_search.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Multimodal vector search

<!-- TABS -->
## Configure your production system

:::note
If you would like to use the production features 
of SuperDuperDB, then you should set the relevant 
connections and configurations in a configuration 
file. Otherwise you are welcome to use "development" mode 
to get going with SuperDuperDB quickly.
:::

```python
import os

os.makedirs('.superduperdb', exist_ok=True)
os.environ['SUPERDUPERDB_CONFIG'] = '.superduperdb/config.yaml'
```


<Tabs>
    <TabItem value="MongoDB Community" label="MongoDB Community" default>
        ```python
        CFG = '''
        data_backend: mongodb://127.0.0.1:27017/documents
        artifact_store: filesystem://./artifact_store
        cluster:
          cdc:
            strategy: null
            uri: ray://127.0.0.1:20000
          compute:
            uri: ray://127.0.0.1:10001
          vector_search:
            backfill_batch_size: 100
            type: in_memory
            uri: http://127.0.0.1:21000
        '''        
        ```
    </TabItem>
    <TabItem value="MongoDB Atlas" label="MongoDB Atlas" default>
        ```python
        CFG = '''
        artifact_store: filesystem://<path-to-artifact-store>
        cluster: 
            compute: ray://<ray-host>
            cdc:    
                uri: http://<cdc-host>:<cdc-port>
            vector_search:
                uri: http://<vector-search-host>:<vector-search-port>
                type: native
        databackend: mongodb+srv://<user>:<password>@<mongo-host>:27017/documents
        '''        
        ```
    </TabItem>
    <TabItem value="SQLite" label="SQLite" default>
        ```python
        CFG = '''
        artifact_store: filesystem://<path-to-artifact-store>
        cluster: 
            compute: ray://<ray-host>
            cdc:    
                uri: http://<cdc-host>:<cdc-port>
            vector_search:
                uri: http://<vector-search-host>:<vector-search-port>
        databackend: sqlite://<path-to-db>.db
        '''        
        ```
    </TabItem>
    <TabItem value="MySQL" label="MySQL" default>
        ```python
        CFG = '''
        artifact_store: filesystem://<path-to-artifact-store>
        cluster: 
            compute: ray://<ray-host>
            cdc:    
                uri: http://<cdc-host>:<cdc-port>
            vector_search:
                uri: http://<vector-search-host>:<vector-search-port>
        databackend: mysql://<user>:<password>@<host>:<port>/database
        '''        
        ```
    </TabItem>
    <TabItem value="Oracle" label="Oracle" default>
        ```python
        CFG = '''
        artifact_store: filesystem://<path-to-artifact-store>
        cluster: 
            compute: ray://<ray-host>
            cdc:    
                uri: http://<cdc-host>:<cdc-port>
            vector_search:
                uri: http://<vector-search-host>:<vector-search-port>
        databackend: mssql://<user>:<password>@<host>:<port>
        '''        
        ```
    </TabItem>
    <TabItem value="PostgreSQL" label="PostgreSQL" default>
        ```python
        CFG = '''
        artifact_store: filesystem://<path-to-artifact-store>
        cluster: 
            compute: ray://<ray-host>
            cdc:    
                uri: http://<cdc-host>:<cdc-port>
            vector_search:
                uri: http://<vector-search-host>:<vector-search-port>
        databackend: postgres://<user>:<password>@<host>:<port</<database>
        '''        
        ```
    </TabItem>
    <TabItem value="Snowflake" label="Snowflake" default>
        ```python
        CFG = '''
        artifact_store: filesystem://<path-to-artifact-store>
        metadata_store: sqlite://<path-to-sqlite-db>.db
        cluster: 
            compute: ray://<ray-host>
            cdc:    
                uri: http://<cdc-host>:<cdc-port>
            vector_search:
                uri: http://<vector-search-host>:<vector-search-port>
        databackend: snowflake://<user>:<password>@<account>/<database>
        '''        
        ```
    </TabItem>
    <TabItem value="Clickhouse" label="Clickhouse" default>
        ```python
        CFG = '''
        artifact_store: filesystem://<path-to-artifact-store>
        metadata_store: sqlite://<path-to-sqlite-db>.db
        cluster: 
            compute: ray://<ray-host>
            cdc:    
                uri: http://<cdc-host>:<cdc-port>
            vector_search:
                uri: http://<vector-search-host>:<vector-search-port>
        databackend: clickhouse://<user>:<password>@<host>:<port>
        '''        
        ```
    </TabItem>
</Tabs>
```python
with open(os.environ['SUPERDUPERDB_CONFIG'], 'w') as f:
    f.write(CFG)
```

<!-- TABS -->
## Start your cluster

:::note
Starting a SuperDuperDB cluster is useful in production and model development
if you want to enable scalable compute, access to the models by multiple users for collaboration, 
monitoring.

If you don't need this, then it is simpler to start in development mode.
:::


<Tabs>
    <TabItem value="Experimental Cluster" label="Experimental Cluster" default>
        ```python
        !python -m superduperdb local-cluster up        
        ```
    </TabItem>
    <TabItem value="Docker-Compose" label="Docker-Compose" default>
        ```python
        !make build_sandbox
        !make testenv_init        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Connect to SuperDuperDB

:::note
Note that this is only relevant if you are running SuperDuperDB in development mode.
Otherwise refer to "Configuring your production system".
:::


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
        !pip install psycopg2
        from superduperdb import superduper
        
        user = 'postgres'
        password = 'postgres'
        port = 5432
        host = 'localhost'
        database = 'test_db'
        db_uri = f"postgres://{user}:{password}@{host}:{port}/{database}"
        
        db = superduper(db_uri, metadata_store=db_uri.replace('postgres://', 'postgresql://'))        
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

```python
from superduperdb.backends.ibis import dtype

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
        !curl -O https://superduperdb-public-demo.s3.amazonaws.com/videos.zip && unzip videos.zip
        import os
        
        data = [f'videos/{x}' for x in os.listdir('./videos')]
        sample_datapoint = data[-1]
        
        from superduperdb.ext.pillow import pil_image
        chunked_model_datatype = pil_image        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        !curl -O https://superduperdb-public-demo.s3.amazonaws.com/audio.zip && unzip audio.zip
        import os
        
        data = [f'audios/{x}' for x in os.listdir('./audios')]
        sample_datapoint = data[-1]
        chunked_model_datatype = dtype('str')        
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
    <TabItem value="Text" label="Text" default>
        ```python
        datatype = 'str'        
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
```python
from superduperdb import DataType
if datatype and isinstance(datatype, DataType):
    db.apply(datatype)
```

<!-- TABS -->
## Setup tables or collections


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        # Note this is an optional step for MongoDB
        # Users can also work directly with `DataType` if they want to add
        # custom data
        from superduperdb import Schema, DataType
        from superduperdb.backends.mongodb import Collection
        
        table_or_collection = Collection('documents')
        USE_SCHEMA = False
        
        if USE_SCHEMA and isinstance(datatype, DataType):
            schema = Schema(fields={'x': datatype})
            db.apply(schema)        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb.backends.ibis import Table
        from superduperdb import Schema, DataType
        from superduperdb.backends.ibis.field_types import dtype
        
        datatype = "str"
        
        if isinstance(datatype, DataType):
            schema = Schema(identifier="schema", fields={"id": dtype("str"), "x": datatype})
        else:
            schema = Schema(
                identifier="schema", fields={"id": dtype("str"), "x": dtype(datatype)}
            )
        
        table_or_collection = Table('documents', schema=schema)
        
        db.apply(table_or_collection)        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Insert data

In order to create data, we need to create a `Schema` for encoding our special `Datatype` column(s) in the databackend.


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb import Document, DataType
        
        def do_insert(data, schema = None):
            
            if schema is None and (datatype is None or isinstance(datatype, str)):
                data = [Document({'x': x['x'], 'y': x['y']}) for x in data]
                db.execute(table_or_collection.insert_many(data))
            elif schema is None and datatype is not None and isinstance(datatype, DataType):
                data = [Document({'x': datatype(x['x']), 'y': x['y']}) for x in data]
                db.execute(table_or_collection.insert_many(data))
            else:
                data = [Document({'x': x['x'], 'y': x['y']}) for x in data]
                db.execute(table_or_collection.insert_many(data, schema=schema))
        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb import Document
        
        def do_insert(data):
            db.execute(table_or_collection.insert([Document({'id': str(idx), 'x': x['x'], 'y': x['y']}) for idx, x in enumerate(data)]))
        
        ```
    </TabItem>
</Tabs>
```python
do_insert(data[:-len(data) // 4])
```

#### Define the embedding model datatype


<Tabs>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb.components.vector_index import sqlvector
        get_chunking_datatype = lambda shape: sqlvector(shape=(shape,))        
        ```
    </TabItem>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb.components.vector_index import vector
        get_chunking_datatype = lambda shape: vector(shape=(shape,))        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Build simple select queries


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        
        select = table_or_collection.find({})        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        
        select = table_or_collection.to_query()        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Create Model Output Type


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        chunked_model_datatype = None        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb.backends.ibis.field_types import dtype
        chunked_model_datatype = dtype('str')        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Apply a chunker for search

:::note
Note that applying a chunker is ***not*** mandatory for search.
If your data is already chunked (e.g. short text snippets or audio) or if you
are searching through something like images, which can't be chunked, then this
won't be necessary.
:::


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        from superduperdb import objectmodel
        
        CHUNK_SIZE = 200
        
        @objectmodel(flatten=True, model_update_kwargs={'document_embedded': False}, datatype=chunked_model_datatype)
        def chunker(text):
            text = text.split()
            chunks = [' '.join(text[i:i + CHUNK_SIZE]) for i in range(0, len(text), CHUNK_SIZE)]
            return chunks        
        ```
    </TabItem>
    <TabItem value="PDF" label="PDF" default>
        ```python
        !pip install -q "unstructured[pdf]"
        from superduperdb import objectmodel
        from unstructured.partition.pdf import partition_pdf
        import PyPDF2
        
        CHUNK_SIZE = 500
        
        @objectmodel(flatten=True, model_update_kwargs={'document_embedded': False}, datatype=chunked_model_datatype)
        def chunker(pdf_file):
            elements = partition_pdf(pdf_file)
            text = '\n'.join([e.text for e in elements])
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
        from superduperdb import objectmodel, Schema
        
        
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
from superduperdb import Listener

upstream_listener = Listener(
    model=chunker,
    select=select,
    key='x',
)

db.apply(upstream_listener)
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
        
        if not get_chunking_datatype:
            model_dtype =  vector(shape=(384,))
        else:
            model_dtype = get_chunking_datatype(384)
        
        # Load the pre-trained sentence transformer model
        model = SentenceTransformer(
            identifier='all-MiniLM-L6-v2',
            postprocess=lambda x: x.tolist(),
            datatype=model_dtype,
        )        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        from torchvision import transforms
        import torch
        import torch.nn as nn
        import torchvision.models as models
        
        import warnings
        
        # Import custom modules
        from superduperdb.ext.torch import TorchModel, tensor
        
        # Define a series of image transformations using torchvision.transforms.Compose
        t = transforms.Compose([
            transforms.Resize((224, 224)),   # Resize the input image to 224x224 pixels (must same as here)
            transforms.CenterCrop((224, 224)),  # Perform a center crop on the resized image
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the tensor with specified mean and standard deviation
        ])
        
        # Define a preprocess function that applies the defined transformations to an input image
        def preprocess(x):
            try:
                return t(x)
            except Exception as e:
                # If an exception occurs during preprocessing, issue a warning and return a tensor of zeros
                warnings.warn(str(e))
                return torch.zeros(3, 224, 224)
        
        # Load the pre-trained ResNet-50 model from torchvision
        resnet50 = models.resnet50(pretrained=True)
        
        # Extract all layers of the ResNet-50 model except the last one
        modules = list(resnet50.children())[:-1]
        resnet50 = nn.Sequential(*modules)
        
        # Create a TorchModel instance with the ResNet-50 model, preprocessing function, and postprocessing lambda
        model = TorchModel(
            identifier='resnet50',
            preprocess=preprocess,
            object=resnet50,
            postprocess=lambda x: x[:, 0, 0],  # Postprocess by extracting the top-left element of the output tensor
            datatype=tensor(torch.float, shape=(2048,))  # Specify the encoder configuration
        )        
        ```
    </TabItem>
    <TabItem value="Text+Image" label="Text+Image" default>
        ```python
        import clip
        from superduperdb import vector
        from superduperdb.ext.torch import TorchModel
        
        # Load the CLIP model and obtain the preprocessing function
        model, preprocess = clip.load("RN50", device='cpu')
        
        # Define a vector with shape (1024,)
        
        if not get_chunking_datatype:
            e =  vector(shape=(1024,))
        else:
            e = get_chunking_datatype(1024)
        
        
        # Create a TorchModel for text encoding
        compatible_model = TorchModel(
            identifier='clip_text', # Unique identifier for the model
            object=model, # CLIP model
            preprocess=lambda x: clip.tokenize(x)[0],  # Model input preprocessing using CLIP 
            postprocess=lambda x: x.tolist(), # Convert the model output to a list
            datatype=e,  # Vector encoder with shape (1024,)
            forward_method='encode_text', # Use the 'encode_text' method for forward pass 
        )
        
        # Create a TorchModel for visual encoding
        model = TorchModel(
            identifier='clip_image',  # Unique identifier for the model
            object=model.visual,  # Visual part of the CLIP model    
            preprocess=preprocess, # Visual preprocessing using CLIP
            postprocess=lambda x: x.tolist(), # Convert the output to a list 
            datatype=e, # Vector encoder with shape (1024,)
        )        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        !pip install librosa
        import librosa
        import numpy as np
        from superduperdb import ObjectModel
        from superduperdb import vector
        
        def audio_embedding(audio_file):
            # Load the audio file
            MAX_SIZE= 10000
            y, sr = librosa.load(audio_file)
            y = y[:MAX_SIZE]
            mfccs = librosa.feature.mfcc(y=y, sr=44000, n_mfcc=1)
            mfccs =  mfccs.squeeze().tolist()
            return mfccs
        
        if not get_chunking_datatype:
            e =  vector(shape=(1000,))
        else:
            e = get_chunking_datatype(1000)
        
        model= ObjectModel(identifier='my-model-audio', object=audio_embedding, datatype=e)        
        ```
    </TabItem>
</Tabs>
## Select outputs of upstream listener

:::note
This is useful if you have performed a first step, such as pre-computing 
features, or chunking your data. You can use this query to 
operate on those outputs.
:::


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb.backends.mongodb import Collection
        
        indexing_key = upstream_listener.outputs
        select = Collection(upstream_listener.outputs).find()        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        indexing_key = upstream_listener.outputs_key
        select = db.load("table", upstream_listener.outputs).to_query()        
        ```
    </TabItem>
</Tabs>
Depending on whether we have chunked the data, 
the indexing key will be different:


<Tabs>
    <TabItem value="Chunked Search" label="Chunked Search" default>
        ```python
        compatible_key = None
        if compatible_model:
            compatible_key = 'y'        
        ```
    </TabItem>
    <TabItem value="Un-chunked Search" label="Un-chunked Search" default>
        ```python
        indexing_key = 'x'
        compatible_key = None
        if compatible_model:
            compatible_key = 'y'        
        ```
    </TabItem>
</Tabs>
## Create vector-index

```python
vector_index_name = 'my-vector-index'
```


<Tabs>
    <TabItem value="1-Modality" label="1-Modality" default>
        ```python
        from superduperdb import VectorIndex, Listener
        
        jobs, _ = db.add(
            VectorIndex(
                vector_index_name,
                indexing_listener=Listener(
                    key=indexing_key,      # the `Document` key `model` should ingest to create embedding
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
        
        jobs, _ = db.add(
            VectorIndex(
                vector_index_name,
                indexing_listener=Listener(
                    key=indexing_key,      # the `Document` key `model` should ingest to create embedding
                    select=select,       # a `Select` query telling which data to search over
                    model=model,         # a `_Predictor` how to convert data to embeddings
                ),
                compatible_listener=Listener(
                    key=compatible_key,      # the `Document` key `model` should ingest to create embedding
                    model=compatible_model,         # a `_Predictor` how to convert data to embeddings
                    active=False,
                    select=None,
                )
            )
        )        
        ```
    </TabItem>
</Tabs>
```python
query_table_or_collection = select.table_or_collection
```

<!-- TABS -->
## Perform a vector search

```python
from superduperdb import Document

def get_sample_item(key, sample_datapoint, datatype=None):
    if not isinstance(datatype, DataType):
        item = Document({key: sample_datapoint})
    else:
        item = Document({key: datatype(sample_datapoint)})

    return item

if compatible_key:
    item = get_sample_item(compatible_key, sample_datapoint, None)
else:
    item = get_sample_item(indexing_key, sample_datapoint, datatype=datatype)
```

Once we have this search target, we can execute a search as follows:


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        select = query_table_or_collection.like(item, vector_index=vector_index_name, n=10).find()        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        select = query_table_or_collection.like(item, vector_index=vector_index_name, n=10).limit(10)        
        ```
    </TabItem>
</Tabs>
```python
results = db.execute(select)
```

<!-- TABS -->
## Visualize Results


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        from IPython.display import Markdown, display
        
        def visualize(item, source):
            display(Markdown(item))
            
        def show(results, output_key, get_original_callable=None):
            for result in results:
                source = None
                if '_source' in result:
                    
                    source = get_original_callable(result['_source'])
                visualize(result[output_key], source)        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        from IPython.display import display
        
        def visualize(item, source):
            display(item)        # item is a PIL.Image
        
        def show(results, output_key, get_original_callable=None):
            for result in results:
                source = None
                if '_source' in result:
                    source = get_original_callable(result['_source'])
                visualize(result[output_key], source)        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        from IPython.display import Audio, display
        
        def visualize(item, source):
            display(Audio(item[1], fs=item[0]))
        
        def show(results, output_key, get_original_callable=None):
            for result in results:
                source = None
                if '_source' in result:
                    
                    source = get_original_callable(result['_source'])
                visualize(result[output_key], source)        
        ```
    </TabItem>
    <TabItem value="PDF" label="PDF" default>
        ```python
        from IPython.display import IFrame, display
        
        def visualize(item, source):
            display(item)
        
        
        def show(results, output_key, get_original_callable=None):
            for result in results:
                source = None
                if '_source' in result:
                    
                    source = get_original_callable(result['_source'])
                visualize(result[output_key], source)        
        ```
    </TabItem>
    <TabItem value="Video" label="Video" default>
        ```python
        from IPython.display import display, HTML
        
        def visualize(uri, source):
            timestamp = source    # increment to the frame you want to start at
            
            # Create HTML code for the video player with a specified source and controls
            video_html = f"""
            <video width="640" height="480" controls>
                <source src="{uri}" type="video/mp4">
            </video>
            <script>
                // Get the video element
                var video = document.querySelector('video');
                
                // Set the current time of the video to the specified timestamp
                video.currentTime = {timestamp};
                
                // Play the video automatically
                video.play();
            </script>
            """
            
            display(HTML(video_html))
        
        
        def show(results, output_key, get_original_callable=None):
            # show only the first video
            for result in results:
                result = result[output_key]
                timestamp = result['current_timestamp']
                source = result['_source']
                uri = get_original_callable(source)['x']
                visualize(uri, timestamp)
                break        
        ```
    </TabItem>
</Tabs>
If your use-case involved chunking, you will want to be able to recover original rows/ documents, 
after getting the result of a vector-search:


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        def get_original(_source):
            return db.execute(table_or_collection.find_one({'_id': _source}))
            
        visualization_key = upstream_listener.outputs        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        def get_original(_source):
            return next(db.execute(table_or_collection.select_using_ids([_source])))
            
        visualization_key = indexing_key        
        ```
    </TabItem>
</Tabs>
```python
show(results, visualization_key, get_original)
```

## Check the system stays updated


<Tabs>
    <TabItem value="Development" label="Development" default>
        ```python
        
        do_insert(data[-len(data) // 4:])        
        ```
    </TabItem>
    <TabItem value="Cluster" label="Cluster" default>
        ```python
        
        # As an example with MongoDB, we show that inserting to/ updating the DB with a different client (potentially from different source)
        # still means that the system stays up-to-date. This should work with any Cluster mode compatible DB (see "Configuring your production system")
        
        collection = pymongo.MongoClient('mongodb://<mongo-host>:/27017/<database>')['<database>'].documents
        collection.insert_many([{'x': x} for x in data[-len(data) // 4:])        
        ```
    </TabItem>
</Tabs>
<DownloadButton filename="multimodal_vector_search.md" />