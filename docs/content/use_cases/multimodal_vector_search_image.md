---
sidebar_label: Multimodal vector search - Image
filename: multimodal_vector_search_image.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Multimodal vector search - Image

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
    <TabItem value="Image" label="Image" default>
        ```python
        !curl -O https://superduperdb-public-demo.s3.amazonaws.com/images.zip && unzip images.zip
        import os
        from PIL import Image
        
        data = [f'images/{x}' for x in os.listdir('./images') if x.endswith(".png")][:200]
        data = [ Image.open(path) for path in data]        
        ```
    </TabItem>
</Tabs>
```python
datas = [{'img': d} for d in data[:100]]
```

<!-- TABS -->
## Insert simple data

After turning on auto_schema, we can directly insert data, and superduperdb will automatically analyze the data type, and match the construction of the table and datatype.

```python
from superduperdb import Document

table_or_collection = db['documents']

ids = db.execute(table_or_collection.insert([Document(data) for data in datas]))
select = table_or_collection.select()
```

<!-- TABS -->
## Build multimodal embedding models

Some embedding models such as [CLIP](https://github.com/openai/CLIP) come in pairs of `model` and `compatible_model`.
Otherwise:


<Tabs>
    <TabItem value="Text-Image" label="Text-Image" default>
        ```python
        !pip install git+https://github.com/openai/CLIP.git
        import clip
        from superduperdb import vector
        from superduperdb.ext.torch import TorchModel
        
        # Load the CLIP model and obtain the preprocessing function
        model, preprocess = clip.load("RN50", device='cpu')
        
        # Define a vector with shape (1024,)
        
        output_datatpye = vector(shape=(1024,))
        
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
indexing_key = 'img' # we use img key for img embedding
compatible_key = 'text' # we use text key for text embedding
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

We can perform the vector searches using two types of data:

- Text: By text description, we can find images similar to the text description.
- Image: By using an image, we can find images similar to the provided image.


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        item = Document({compatible_key: "Find a black dog"})        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        from IPython.display import display
        search_image = data[0]
        display(search_image)
        item = Document({indexing_key: search_image})        
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
    display(result[indexing_key])
```

## Check the system stays updated

You can add new data; once the data is added, all related models will perform calculations according to the underlying constructed model and listener, simultaneously updating the vector index to ensure that each query uses the latest data.

```python
new_datas = [{'img': d} for d in data[100:200]]
ids = db.execute(table_or_collection.insert(new_datas))
```

<DownloadButton filename="multimodal_vector_search_-_image.md" />
