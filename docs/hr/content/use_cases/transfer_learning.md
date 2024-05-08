---
sidebar_label: Transfer learning
filename: transfer_learning.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Transfer learning

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
    <TabItem value="labeled_text" label="labeled_text" default>
        ```python
        !curl -O https://superduperdb-public-demo.s3.amazonaws.com/text_classification.json
        import json
        
        with open("text_classification.json", "r") as f:
            data = json.load(f)
        sample_datapoint = data[-1]        
        ```
    </TabItem>
    <TabItem value="labeled_image" label="labeled_image" default>
        ```python
        !curl -O https://superduperdb-public-demo.s3.amazonaws.com/images_classification.zip && unzip images.zip
        import json
        from PIL import Image
        
        with open('images/images.json', 'r') as f:
            data = json.load(f)
        
        data = [{'x': Image.open(d['image_path']), 'y': d['label']} for d in data]
        sample_datapoint = data[-1]        
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
                data = [Document({'x': x['x'], 'y': x['y']}) if isinstance(x, dict) and 'x' in x and 'y' in x else Document({'x': x}) for x in data]
                db.execute(table_or_collection.insert_many(data))
            elif schema is None and datatype is not None and isinstance(datatype, DataType):
                data = [Document({'x': datatype(x['x']), 'y': x['y']}) if isinstance(x, dict) and 'x' in x and 'y' in x else Document({'x': datatype(x)}) for x in data]
                db.execute(table_or_collection.insert_many(data))
            else:
                data = [Document({'x': x['x'], 'y': x['y']}) if isinstance(x, dict) and 'x' in x and 'y' in x else Document({'x': x}) for x in data]
                db.execute(table_or_collection.insert_many(data, schema=schema))
        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb import Document
        
        def do_insert(data):
            db.execute(table_or_collection.insert([Document({'id': str(idx), 'x': x['x'], 'y': x['y']}) if isinstance(x, dict) and 'x' in x and 'y' in x else Document({'id': str(idx), 'x': x}) for idx, x in enumerate(data)]))
        
        ```
    </TabItem>
</Tabs>
```python
do_insert(data[:-len(data) // 4])
```

<!-- TABS -->
## Compute features


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        
        key = 'txt'
        
        import sentence_transformers
        from superduperdb import vector, Listener
        from superduperdb.ext.sentence_transformers import SentenceTransformer
        
        superdupermodel = SentenceTransformer(
            identifier="embedding",
            object=sentence_transformers.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
            datatype=vector(shape=(384,)),
            postprocess=lambda x: x.tolist(),
        )
        
        jobs, listener = db.apply(
            Listener(
                model=superdupermodel,
                select=select,
                key=key,
                identifier="features"
            )
        )        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        
        key = 'image'
        
        import torchvision.models as models
        from torchvision import transforms
        from superduperdb.ext.torch import TorchModel
        from superduperdb import Listener
        from PIL import Image
        
        class TorchVisionEmbedding:
            def __init__(self):
                # Load the pre-trained ResNet-18 model
                self.resnet = models.resnet18(pretrained=True)
                
                # Set the model to evaluation mode
                self.resnet.eval()
                
            def preprocess(self, image_array):
                # Preprocess the image
                image = Image.fromarray(image_array.astype(np.uint8))
                preprocess = preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                tensor_image = preprocess(image)
                return tensor_image
                
        model = TorchVisionEmbedding()
        superdupermodel = TorchModel(identifier='my-vision-model-torch', object=model.resnet, preprocess=model.preprocess, postprocess=lambda x: x.numpy().tolist())
        
        jobs, listener = db.apply(
            Listener(
                model=superdupermodel,
                select=select,
                key=key,
                identifier="features"
            )
        )        
        ```
    </TabItem>
</Tabs>
## Choose input key from listener outputs

:::note
This is useful if you have performed a first step, such as pre-computing 
features, or chunking your data. You can use this query to 
choose the input key for further models such as classification models.
:::


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        input_key = listener.outputs
        select = table_or_collection.find()        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        input_key = listener.outputs
        select = table_or_collection.outputs(listener.predict_id).select('y', input_key)
        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Build and train classifier


<Tabs>
    <TabItem value="Scikit-Learn" label="Scikit-Learn" default>
        ```python
        from sklearn.linear_model import LogisticRegression
        from superduperdb.ext.sklearn.model import SklearnTrainer, Estimator
        
        # Create a Logistic Regression model
        model = LogisticRegression()
        model = Estimator(
            object=model,
            identifier='my-model',
            trainer=SklearnTrainer(
                key=(input_key, 'y'),
                select=select,
            )
        )        
        ```
    </TabItem>
    <TabItem value="Torch" label="Torch" default>
        ```python
        from torch import nn
        from superduperdb.ext.torch.model import TorchModel
        from superduperdb.ext.torch.training import TorchTrainer
        
        
        class SimpleModel(nn.Module):
            def __init__(self, input_size=16, hidden_size=32, num_classes=3):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, num_classes)
        
            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out
        
        # Loss function
        def my_loss(X, y):
            return torch.nn.functional.binary_cross_entropy_with_logits(
                X[:, 0], y.type(torch.float)
            )
        
        
        # Create a Logistic Regression model
        model = SimpleModel()
        model = TorchModel(
            identifier='my-model',
            object=model,         
            trainer=TorchTrainer(
                key=(input_key, 'y'),
                identifier='my_trainer',
                objective=my_loss,
                loader_kwargs={'batch_size': 10},
                max_iterations=100,
                validation_interval=10,
                select=select,
            ),
        )        
        ```
    </TabItem>
</Tabs>
The following command adds the model to the system and trains the model in one command.

```python
db.apply(model)
```

<DownloadButton filename="transfer_learning.md" />
