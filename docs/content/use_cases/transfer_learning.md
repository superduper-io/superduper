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
of Superduper, then you should set the relevant 
connections and configurations in a configuration 
file. Otherwise you are welcome to use "development" mode 
to get going with Superduper quickly.
:::

```python
import os

os.makedirs('.superduper', exist_ok=True)
os.environ['SUPERDUPER_CONFIG'] = '.superduper/config.yaml'
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
with open(os.environ['SUPERDUPER_CONFIG'], 'w') as f:
    f.write(CFG)
```

<!-- TABS -->
## Start your cluster

:::note
Starting a Superduper cluster is useful in production and model development
if you want to enable scalable compute, access to the models by multiple users for collaboration, 
monitoring.

If you don't need this, then it is simpler to start in development mode.
:::


<Tabs>
    <TabItem value="Experimental Cluster" label="Experimental Cluster" default>
        ```python
        !python -m superduper local-cluster up        
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
## Connect to Superduper

:::note
Note that this is only relevant if you are running Superduper in development mode.
Otherwise refer to "Configuring your production system".
:::


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduper import superduper
        
        db = superduper('mongodb://localhost:27017/documents')        
        ```
    </TabItem>
    <TabItem value="SQLite" label="SQLite" default>
        ```python
        from superduper import superduper
        db = superduper('sqlite://my_db.db')        
        ```
    </TabItem>
    <TabItem value="MySQL" label="MySQL" default>
        ```python
        from superduper import superduper
        
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
        from superduper import superduper
        
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
        from superduper import superduper
        
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
        from superduper import superduper
        
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
        from superduper import superduper
        
        user = 'default'
        password = ''
        port = 8123
        host = 'localhost'
        
        db = superduper(f"clickhouse://{user}:{password}@{host}:{port}", metadata_store=f'mongomock://meta')        
        ```
    </TabItem>
    <TabItem value="DuckDB" label="DuckDB" default>
        ```python
        from superduper import superduper
        
        db = superduper('duckdb://mydb.duckdb')        
        ```
    </TabItem>
    <TabItem value="Pandas" label="Pandas" default>
        ```python
        from superduper import superduper
        
        db = superduper(['my.csv'], metadata_store=f'mongomock://meta')        
        ```
    </TabItem>
    <TabItem value="MongoMock" label="MongoMock" default>
        ```python
        from superduper import superduper
        
        db = superduper('mongomock:///test_db')        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Get useful sample data


<Tabs>
    <TabItem value="Text-Classification" label="Text-Classification" default>
        ```python
        !curl -O https://superduper-public-demo.s3.amazonaws.com/text_classification.json
        import json
        
        with open("text_classification.json", "r") as f:
            data = json.load(f)
        num_classes = 2        
        ```
    </TabItem>
    <TabItem value="Image-Classification" label="Image-Classification" default>
        ```python
        !curl -O https://superduper-public-demo.s3.amazonaws.com/images_classification.zip && unzip images_classification.zip
        import json
        from PIL import Image
        
        with open('images/images.json', 'r') as f:
            data = json.load(f)
        
        data = [{'x': Image.open(d['image_path']), 'y': d['label']} for d in data]
        num_classes = 2        
        ```
    </TabItem>
</Tabs>
After obtaining the data, we insert it into the database.


<Tabs>
    <TabItem value="Text-Classification" label="Text-Classification" default>
        ```python
        datas = [{'txt': d['x'], 'label': d['y']} for d in data]        
        ```
    </TabItem>
    <TabItem value="Image-Classification" label="Image-Classification" default>
        ```python
        datas = [{'image': d['x'], 'label': d['y']} for d in data]        
        ```
    </TabItem>
</Tabs>
<!-- TABS -->
## Insert simple data

After turning on auto_schema, we can directly insert data, and Superduper will automatically analyze the data type, and match the construction of the table and datatype.

```python
from superduper import Document

table_or_collection = db['documents']

ids = db.execute(table_or_collection.insert([Document(data) for data in datas]))
select = table_or_collection.select()
```

<!-- TABS -->
## Compute features


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        key = 'txt'
        import sentence_transformers
        from superduper import vector, Listener
        from superduper.ext.sentence_transformers import SentenceTransformer
        
        superdupermodel = SentenceTransformer(
            identifier="embedding",
            object=sentence_transformers.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
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
        from superduper.ext.torch import TorchModel
        from superduper import Listener
        from PIL import Image
        
        class TorchVisionEmbedding:
            def __init__(self):
                # Load the pre-trained ResNet-18 model
                self.resnet = models.resnet18(pretrained=True)
                
                # Set the model to evaluation mode
                self.resnet.eval()
                
            def preprocess(self, image):
                # Preprocess the image
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
## Choose features key from feature listener


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        input_key = listener.outputs
        training_select = select        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        input_key = listener.outputs
        training_select = select.outputs(listener.predict_id)        
        ```
    </TabItem>
</Tabs>
We can find the calculated feature data from the database.

```python
feature = list(training_select.limit(1).execute())[0][input_key]
feature_size = len(feature)
```

<!-- TABS -->
## Build and train classifier


<Tabs>
    <TabItem value="Scikit-Learn" label="Scikit-Learn" default>
        ```python
        from superduper.ext.sklearn import Estimator, SklearnTrainer
        from sklearn.svm import SVC
        
        model = Estimator(
            identifier="my-model",
            object=SVC(),
            trainer=SklearnTrainer(
                "my-trainer",
                key=(input_key, "label"),
                select=training_select,
            ),
        )        
        ```
    </TabItem>
    <TabItem value="Torch" label="Torch" default>
        ```python
        import torch
        from torch import nn
        from superduper.ext.torch.model import TorchModel
        from superduper.ext.torch.training import TorchTrainer
        from torch.nn.functional import cross_entropy
        
        
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
        
        preprocess = lambda x: torch.tensor(x)
        
        # Postprocess function for the model output    
        def postprocess(x):
            return int(x.topk(1)[1].item())
        
        def data_transform(features, label):
            return torch.tensor(features), label
        
        # Create a Logistic Regression model
        # feature_length is the input feature size
        model = SimpleModel(feature_size, num_classes=num_classes)
        model = TorchModel(
            identifier='my-model',
            object=model,         
            preprocess=preprocess,
            postprocess=postprocess,
            trainer=TorchTrainer(
                key=(input_key, 'label'),
                identifier='my_trainer',
                objective=cross_entropy,
                loader_kwargs={'batch_size': 10},
                max_iterations=1000,
                validation_interval=100,
                select=select,
                transform=data_transform,
            ),
        )        
        ```
    </TabItem>
</Tabs>
Define a validation for evaluating the effect after training.

```python
from superduper import Dataset, Metric, Validation

def acc(x, y):
    return sum([xx == yy for xx, yy in zip(x, y)]) / len(x)


accuracy = Metric(identifier="acc", object=acc)
validation = Validation(
    "transfer_learning_performance",
    key=(input_key, "label"),
    datasets=[
        Dataset(identifier="my-valid", select=training_select.add_fold('valid'))
    ],
    metrics=[accuracy],
)
model.validation = validation
```

If we execute the apply function, then the model will be added to the database, and because the model has a Trainer, it will perform training tasks.

```python
db.apply(model)
```

Get the training metrics


<Tabs>
    <TabItem value="Scikit-Learn" label="Scikit-Learn" default>
        ```python
        # Load the model from the database
        model = db.load('model', model.identifier)
        model.metric_values        
        ```
    </TabItem>
    <TabItem value="Torch" label="Torch" default>
        ```python
        !pip -q install matplotlib
        from matplotlib import pyplot as plt
        
        # Load the model from the database
        model = db.load('model', model.identifier)
        
        # Plot the accuracy values
        plt.plot(model.trainer.metric_values['my-valid/acc'])
        plt.show()        
        ```
    </TabItem>
</Tabs>
<DownloadButton filename="transfer_learning.md" />
