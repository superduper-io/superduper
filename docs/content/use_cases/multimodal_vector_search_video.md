---
sidebar_label: Multimodal vector search - Video
filename: multimodal_vector_search_video.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Multimodal vector search - Video

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


<Tabs>
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
</Tabs>
```python
datas = [{'x': d} for d in data[:3]]
```

<!-- TABS -->
## Create datatype

Data types such as "text" or "integer" which are natively support by your `db.databackend` don't need a datatype.

Otherwise do one of the following:


<Tabs>
    <TabItem value="Video" label="Video" default>
        ```python
        from superduperdb import DataType
        
        # Create an instance of the Encoder with the identifier 'video_on_file' and load_hybrid set to False
        datatype = DataType(
            identifier='video_on_file',
            encodable='file',
        )        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Setup tables or collections

```python
from superduperdb.components.table import Table
from superduperdb import Schema

schema = Schema(identifier="schema", fields={"x": datatype})
table_or_collection = Table("documents", schema=schema)
db.apply(table_or_collection)
```

Inserting data, all fields will be matched with the schema for data conversion.

```python
db['documents'].insert(datas).execute()
select = db['documents'].select()
```

<!-- TABS -->
## Apply a chunker for search

:::note
Note that applying a chunker is ***not*** mandatory for search.
If your data is already chunked (e.g. short text snippets or audio) or if you
are searching through something like images, which can't be chunked, then this
won't be necessary.
:::


<Tabs>
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
</Tabs>
Now we apply this chunker to the data by wrapping the chunker in `Listener`:

```python
from superduperdb import Listener

upstream_listener = Listener(
    model=chunker,
    select=select,
    key='x',
    uuid="chunk",
)

db.apply(upstream_listener)
```

## Build multimodal embedding models

We define the output data type of a model as a vector for vector transformation.


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb.components.vector_index import vector
        output_datatpye = vector(shape=(1024,))        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb.components.vector_index import sqlvector
        output_datatpye = sqlvector(shape=(1024,))        
        ```
    </TabItem>
</Tabs>
Then define two models, one for text embedding and one for image embedding.


<Tabs>
    <TabItem value="Text-Image" label="Text-Image" default>
        ```python
        !pip install git+https://github.com/openai/CLIP.git
        import clip
        from superduperdb import vector
        from superduperdb.ext.torch import TorchModel
        
        # Load the CLIP model and obtain the preprocessing function
        model, preprocess = clip.load("ViT-B/32", device='cpu')
        
        # Create a TorchModel for text encoding
        compatible_model = TorchModel(
            identifier='clip_text', # Unique identifier for the model
            object=model, # CLIP model
            preprocess=lambda x: clip.tokenize(x)[0],  # Model input preprocessing using CLIP 
            postprocess=lambda x: x.tolist(), # Convert the model output to a list
            datatype=output_datatpye,  # Vector encoder with shape (1024,)
            forward_method='encode_text', # Use the 'encode_text' method for forward pass 
        )
        
        # Create a TorchModel for visual encoding
        model = TorchModel(
            identifier='clip_image',  # Unique identifier for the model
            object=model.visual,  # Visual part of the CLIP model    
            preprocess=preprocess, # Visual preprocessing using CLIP
            postprocess=lambda x: x.tolist(), # Convert the output to a list 
            datatype=output_datatpye, # Vector encoder with shape (1024,)
        )        
        ```
    </TabItem>
</Tabs>
Because we use multimodal models, we define different keys to specify which model to use for embedding calculations in the vector_index.

```python
compatible_key = 'text' # we use text key for text embedding
indexing_key = upstream_listener.outputs_key + '.image' # we use indexing_key for image embedding, use the image field of the result
select = upstream_listener.outputs_select
```

## Create vector-index

```python
vector_index_name = 'my-vector-index'
```


<Tabs>
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

## Perform a vector search

We can perform the vector searches using text description:


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        from superduperdb import Document
        item = Document({compatible_key: "The moment of a soccer shot"})        
        ```
    </TabItem>
</Tabs>
Once we have this search target, we can execute a search as follows.

```python
select = query_table_or_collection.like(item, vector_index=vector_index_name, n=5).select()
results = list(db.execute(select))
```

## Visualize Results

```python
from IPython.display import display
for result in results:
    display(Document(result.unpack())[indexing_key])
```

## Check the system stays updated

You can add new data; once the data is added, all related models will perform calculations according to the underlying constructed model and listener, simultaneously updating the vector index to ensure that each query uses the latest data.

```python
new_datas = [{'x': data[-1]}]
ids = db['documents'].insert(new_datas).execute()
```

<DownloadButton filename="multimodal_vector_search_-_video.md" />
