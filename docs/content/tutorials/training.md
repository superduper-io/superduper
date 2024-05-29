
# Training and Managing MNIST Predictions with SuperDuperDB

:::note
This tutorial guides you through the implementation of a classic machine learning task: MNIST handwritten digit recognition. The twist? We perform the task directly on data hosted in a database using SuperDuperDB.
:::

This example makes it easy to connect any of your image recognition models directly to your database in real-time. 

```python
!pip install torch torchvision
```

<details>
<summary>Outputs</summary>

</details>

First, we need to establish a connection to a MongoDB datastore via SuperDuperDB. 

```python
from superduperdb import superduper
    
db = superduper('mongomock://')
```

<details>
<summary>Outputs</summary>
<pre>
    2024-May-24 16:42:04.28| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.base.build:69   | Data Client is ready. mongomock.MongoClient('localhost', 27017)
    2024-May-24 16:42:04.30| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.base.build:42   | Connecting to Metadata Client with engine:  mongomock.MongoClient('localhost', 27017)
    2024-May-24 16:42:04.30| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.base.build:155  | Connecting to compute client: None
    2024-May-24 16:42:04.30| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.base.datalayer:85   | Building Data Layer
    2024-May-24 16:42:04.30| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.base.build:220  | Configuration: 
     +---------------+--------------+
    | Configuration |    Value     |
    +---------------+--------------+
    |  Data Backend | mongomock:// |
    +---------------+--------------+

</pre>
</details>

After establishing a connection to MongoDB, the next step is to load the MNIST dataset. SuperDuperDB's strength lies in handling diverse data types, especially those that are not supported by standard databases. To achieve this, we use an `Encoder` in conjunction with `Document` wrappers. These components allow Python dictionaries containing non-JSONable or bytes objects to be seamlessly inserted into the underlying data infrastructure.

```python
import torchvision
from superduperdb import Document

import random

# Load MNIST images as Python objects using the Python Imaging Library.
# Each MNIST item is a tuple (image, label)
mnist_data = list(torchvision.datasets.MNIST(root='./data', download=True))

document_list = [Document({'img': x[0], 'class': x[1]}) for x in mnist_data]

# Shuffle the data and select a subset of 1000 documents
random.shuffle(document_list)
data = document_list[:1000]

# Insert the selected data into the mnist_collection which we mentioned before like: mnist_collection = Collection('mnist')
db['mnist'].insert_many(data[:-100]).execute()
```

<details>
<summary>Outputs</summary>
<pre>
    2024-May-24 16:42:05.79| WARNING  | Duncans-MacBook-Pro.fritz.box| superduperdb.misc.annotations:117  | add is deprecated and will be removed in a future release.
    2024-May-24 16:42:05.79| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.components.component:386  | Initializing DataType : dill
    2024-May-24 16:42:05.79| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.components.component:389  | Initialized  DataType : dill successfully
    2024-May-24 16:42:05.79| WARNING  | Duncans-MacBook-Pro.fritz.box| superduperdb.backends.local.artifacts:82   | File /tmp/e6eb888f0b0fbbab905029cb309537b9383919a6 already exists
    2024-May-24 16:42:05.79| WARNING  | Duncans-MacBook-Pro.fritz.box| superduperdb.backends.local.artifacts:82   | File /tmp/ee1a946181f065af29a3c8637b2858b153d8fc8e already exists
    2024-May-24 16:42:05.79| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.components.component:386  | Initializing DataType : pil_image
    2024-May-24 16:42:05.79| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.components.component:389  | Initialized  DataType : pil_image successfully
    2024-May-24 16:42:05.94| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function callable_job at 0x110261c60\>
    2024-May-24 16:42:06.06| SUCCESS  | Duncans-MacBook-Pro.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x151f9e8d0\>.  function:\<function callable_job at 0x110261c60\> future:77cca682-2f87-4374-8f3a-097430cfc3b5

</pre>
<pre>
    ([ObjectId('6650a73d310636ceeea3628d'),
      ObjectId('6650a73d310636ceeea3628e'),
      ObjectId('6650a73d310636ceeea3628f'),
      ObjectId('6650a73d310636ceeea36290'),
      ObjectId('6650a73d310636ceeea36291'),
      ObjectId('6650a73d310636ceeea36292'),
      ObjectId('6650a73d310636ceeea36293'),
      ObjectId('6650a73d310636ceeea36294'),
      ObjectId('6650a73d310636ceeea36295'),
      ObjectId('6650a73d310636ceeea36296'),
      ObjectId('6650a73d310636ceeea36297'),
      ObjectId('6650a73d310636ceeea36298'),
      ObjectId('6650a73d310636ceeea36299'),
      ObjectId('6650a73d310636ceeea3629a'),
      ObjectId('6650a73d310636ceeea3629b'),
      ObjectId('6650a73d310636ceeea3629c'),
      ObjectId('6650a73d310636ceeea3629d'),
      ObjectId('6650a73d310636ceeea3629e'),
      ObjectId('6650a73d310636ceeea3629f'),
      ObjectId('6650a73d310636ceeea362a0'),
      ObjectId('6650a73d310636ceeea362a1'),
      ObjectId('6650a73d310636ceeea362a2'),
      ObjectId('6650a73d310636ceeea362a3'),
      ObjectId('6650a73d310636ceeea362a4'),
      ObjectId('6650a73d310636ceeea362a5'),
      ObjectId('6650a73d310636ceeea362a6'),
      ObjectId('6650a73d310636ceeea362a7'),
      ObjectId('6650a73d310636ceeea362a8'),
      ObjectId('6650a73d310636ceeea362a9'),
      ObjectId('6650a73d310636ceeea362aa'),
      ObjectId('6650a73d310636ceeea362ab'),
      ObjectId('6650a73d310636ceeea362ac'),
      ObjectId('6650a73d310636ceeea362ad'),
      ObjectId('6650a73d310636ceeea362ae'),
      ObjectId('6650a73d310636ceeea362af'),
      ObjectId('6650a73d310636ceeea362b0'),
      ObjectId('6650a73d310636ceeea362b1'),
      ObjectId('6650a73d310636ceeea362b2'),
      ObjectId('6650a73d310636ceeea362b3'),
      ObjectId('6650a73d310636ceeea362b4'),
      ObjectId('6650a73d310636ceeea362b5'),
      ObjectId('6650a73d310636ceeea362b6'),
      ObjectId('6650a73d310636ceeea362b7'),
      ObjectId('6650a73d310636ceeea362b8'),
      ObjectId('6650a73d310636ceeea362b9'),
      ObjectId('6650a73d310636ceeea362ba'),
      ObjectId('6650a73d310636ceeea362bb'),
      ObjectId('6650a73d310636ceeea362bc'),
      ObjectId('6650a73d310636ceeea362bd'),
      ObjectId('6650a73d310636ceeea362be'),
      ObjectId('6650a73d310636ceeea362bf'),
      ObjectId('6650a73d310636ceeea362c0'),
      ObjectId('6650a73d310636ceeea362c1'),
      ObjectId('6650a73d310636ceeea362c2'),
      ObjectId('6650a73d310636ceeea362c3'),
      ObjectId('6650a73d310636ceeea362c4'),
      ObjectId('6650a73d310636ceeea362c5'),
      ObjectId('6650a73d310636ceeea362c6'),
      ObjectId('6650a73d310636ceeea362c7'),
      ObjectId('6650a73d310636ceeea362c8'),
      ObjectId('6650a73d310636ceeea362c9'),
      ObjectId('6650a73d310636ceeea362ca'),
      ObjectId('6650a73d310636ceeea362cb'),
      ObjectId('6650a73d310636ceeea362cc'),
      ObjectId('6650a73d310636ceeea362cd'),
      ObjectId('6650a73d310636ceeea362ce'),
      ObjectId('6650a73d310636ceeea362cf'),
      ObjectId('6650a73d310636ceeea362d0'),
      ObjectId('6650a73d310636ceeea362d1'),
      ObjectId('6650a73d310636ceeea362d2'),
      ObjectId('6650a73d310636ceeea362d3'),
      ObjectId('6650a73d310636ceeea362d4'),
      ObjectId('6650a73d310636ceeea362d5'),
      ObjectId('6650a73d310636ceeea362d6'),
      ObjectId('6650a73d310636ceeea362d7'),
      ObjectId('6650a73d310636ceeea362d8'),
      ObjectId('6650a73d310636ceeea362d9'),
      ObjectId('6650a73d310636ceeea362da'),
      ObjectId('6650a73d310636ceeea362db'),
      ObjectId('6650a73d310636ceeea362dc'),
      ObjectId('6650a73d310636ceeea362dd'),
      ObjectId('6650a73d310636ceeea362de'),
      ObjectId('6650a73d310636ceeea362df'),
      ObjectId('6650a73d310636ceeea362e0'),
      ObjectId('6650a73d310636ceeea362e1'),
      ObjectId('6650a73d310636ceeea362e2'),
      ObjectId('6650a73d310636ceeea362e3'),
      ObjectId('6650a73d310636ceeea362e4'),
      ObjectId('6650a73d310636ceeea362e5'),
      ObjectId('6650a73d310636ceeea362e6'),
      ObjectId('6650a73d310636ceeea362e7'),
      ObjectId('6650a73d310636ceeea362e8'),
      ObjectId('6650a73d310636ceeea362e9'),
      ObjectId('6650a73d310636ceeea362ea'),
      ObjectId('6650a73d310636ceeea362eb'),
      ObjectId('6650a73d310636ceeea362ec'),
      ObjectId('6650a73d310636ceeea362ed'),
      ObjectId('6650a73d310636ceeea362ee'),
      ObjectId('6650a73d310636ceeea362ef'),
      ObjectId('6650a73d310636ceeea362f0'),
      ObjectId('6650a73d310636ceeea362f1'),
      ObjectId('6650a73d310636ceeea362f2'),
      ObjectId('6650a73d310636ceeea362f3'),
      ObjectId('6650a73d310636ceeea362f4'),
      ObjectId('6650a73d310636ceeea362f5'),
      ObjectId('6650a73d310636ceeea362f6'),
      ObjectId('6650a73d310636ceeea362f7'),
      ObjectId('6650a73d310636ceeea362f8'),
      ObjectId('6650a73d310636ceeea362f9'),
      ObjectId('6650a73d310636ceeea362fa'),
      ObjectId('6650a73d310636ceeea362fb'),
      ObjectId('6650a73d310636ceeea362fc'),
      ObjectId('6650a73d310636ceeea362fd'),
      ObjectId('6650a73d310636ceeea362fe'),
      ObjectId('6650a73d310636ceeea362ff'),
      ObjectId('6650a73d310636ceeea36300'),
      ObjectId('6650a73d310636ceeea36301'),
      ObjectId('6650a73d310636ceeea36302'),
      ObjectId('6650a73d310636ceeea36303'),
      ObjectId('6650a73d310636ceeea36304'),
      ObjectId('6650a73d310636ceeea36305'),
      ObjectId('6650a73d310636ceeea36306'),
      ObjectId('6650a73d310636ceeea36307'),
      ObjectId('6650a73d310636ceeea36308'),
      ObjectId('6650a73d310636ceeea36309'),
      ObjectId('6650a73d310636ceeea3630a'),
      ObjectId('6650a73d310636ceeea3630b'),
      ObjectId('6650a73d310636ceeea3630c'),
      ObjectId('6650a73d310636ceeea3630d'),
      ObjectId('6650a73d310636ceeea3630e'),
      ObjectId('6650a73d310636ceeea3630f'),
      ObjectId('6650a73d310636ceeea36310'),
      ObjectId('6650a73d310636ceeea36311'),
      ObjectId('6650a73d310636ceeea36312'),
      ObjectId('6650a73d310636ceeea36313'),
      ObjectId('6650a73d310636ceeea36314'),
      ObjectId('6650a73d310636ceeea36315'),
      ObjectId('6650a73d310636ceeea36316'),
      ObjectId('6650a73d310636ceeea36317'),
      ObjectId('6650a73d310636ceeea36318'),
      ObjectId('6650a73d310636ceeea36319'),
      ObjectId('6650a73d310636ceeea3631a'),
      ObjectId('6650a73d310636ceeea3631b'),
      ObjectId('6650a73d310636ceeea3631c'),
      ObjectId('6650a73d310636ceeea3631d'),
      ObjectId('6650a73d310636ceeea3631e'),
      ObjectId('6650a73d310636ceeea3631f'),
      ObjectId('6650a73d310636ceeea36320'),
      ObjectId('6650a73d310636ceeea36321'),
      ObjectId('6650a73d310636ceeea36322'),
      ObjectId('6650a73d310636ceeea36323'),
      ObjectId('6650a73d310636ceeea36324'),
      ObjectId('6650a73d310636ceeea36325'),
      ObjectId('6650a73d310636ceeea36326'),
      ObjectId('6650a73d310636ceeea36327'),
      ObjectId('6650a73d310636ceeea36328'),
      ObjectId('6650a73d310636ceeea36329'),
      ObjectId('6650a73d310636ceeea3632a'),
      ObjectId('6650a73d310636ceeea3632b'),
      ObjectId('6650a73d310636ceeea3632c'),
      ObjectId('6650a73d310636ceeea3632d'),
      ObjectId('6650a73d310636ceeea3632e'),
      ObjectId('6650a73d310636ceeea3632f'),
      ObjectId('6650a73d310636ceeea36330'),
      ObjectId('6650a73d310636ceeea36331'),
      ObjectId('6650a73d310636ceeea36332'),
      ObjectId('6650a73d310636ceeea36333'),
      ObjectId('6650a73d310636ceeea36334'),
      ObjectId('6650a73d310636ceeea36335'),
      ObjectId('6650a73d310636ceeea36336'),
      ObjectId('6650a73d310636ceeea36337'),
      ObjectId('6650a73d310636ceeea36338'),
      ObjectId('6650a73d310636ceeea36339'),
      ObjectId('6650a73d310636ceeea3633a'),
      ObjectId('6650a73d310636ceeea3633b'),
      ObjectId('6650a73d310636ceeea3633c'),
      ObjectId('6650a73d310636ceeea3633d'),
      ObjectId('6650a73d310636ceeea3633e'),
      ObjectId('6650a73d310636ceeea3633f'),
      ObjectId('6650a73d310636ceeea36340'),
      ObjectId('6650a73d310636ceeea36341'),
      ObjectId('6650a73d310636ceeea36342'),
      ObjectId('6650a73d310636ceeea36343'),
      ObjectId('6650a73d310636ceeea36344'),
      ObjectId('6650a73d310636ceeea36345'),
      ObjectId('6650a73d310636ceeea36346'),
      ObjectId('6650a73d310636ceeea36347'),
      ObjectId('6650a73d310636ceeea36348'),
      ObjectId('6650a73d310636ceeea36349'),
      ObjectId('6650a73d310636ceeea3634a'),
      ObjectId('6650a73d310636ceeea3634b'),
      ObjectId('6650a73d310636ceeea3634c'),
      ObjectId('6650a73d310636ceeea3634d'),
      ObjectId('6650a73d310636ceeea3634e'),
      ObjectId('6650a73d310636ceeea3634f'),
      ObjectId('6650a73d310636ceeea36350'),
      ObjectId('6650a73d310636ceeea36351'),
      ObjectId('6650a73d310636ceeea36352'),
      ObjectId('6650a73d310636ceeea36353'),
      ObjectId('6650a73d310636ceeea36354'),
      ObjectId('6650a73d310636ceeea36355'),
      ObjectId('6650a73d310636ceeea36356'),
      ObjectId('6650a73d310636ceeea36357'),
      ObjectId('6650a73d310636ceeea36358'),
      ObjectId('6650a73d310636ceeea36359'),
      ObjectId('6650a73d310636ceeea3635a'),
      ObjectId('6650a73d310636ceeea3635b'),
      ObjectId('6650a73d310636ceeea3635c'),
      ObjectId('6650a73d310636ceeea3635d'),
      ObjectId('6650a73d310636ceeea3635e'),
      ObjectId('6650a73d310636ceeea3635f'),
      ObjectId('6650a73d310636ceeea36360'),
      ObjectId('6650a73d310636ceeea36361'),
      ObjectId('6650a73d310636ceeea36362'),
      ObjectId('6650a73d310636ceeea36363'),
      ObjectId('6650a73d310636ceeea36364'),
      ObjectId('6650a73d310636ceeea36365'),
      ObjectId('6650a73d310636ceeea36366'),
      ObjectId('6650a73d310636ceeea36367'),
      ObjectId('6650a73d310636ceeea36368'),
      ObjectId('6650a73d310636ceeea36369'),
      ObjectId('6650a73d310636ceeea3636a'),
      ObjectId('6650a73d310636ceeea3636b'),
      ObjectId('6650a73d310636ceeea3636c'),
      ObjectId('6650a73d310636ceeea3636d'),
      ObjectId('6650a73d310636ceeea3636e'),
      ObjectId('6650a73d310636ceeea3636f'),
      ObjectId('6650a73d310636ceeea36370'),
      ObjectId('6650a73d310636ceeea36371'),
      ObjectId('6650a73d310636ceeea36372'),
      ObjectId('6650a73d310636ceeea36373'),
      ObjectId('6650a73d310636ceeea36374'),
      ObjectId('6650a73d310636ceeea36375'),
      ObjectId('6650a73d310636ceeea36376'),
      ObjectId('6650a73d310636ceeea36377'),
      ObjectId('6650a73d310636ceeea36378'),
      ObjectId('6650a73d310636ceeea36379'),
      ObjectId('6650a73d310636ceeea3637a'),
      ObjectId('6650a73d310636ceeea3637b'),
      ObjectId('6650a73d310636ceeea3637c'),
      ObjectId('6650a73d310636ceeea3637d'),
      ObjectId('6650a73d310636ceeea3637e'),
      ObjectId('6650a73d310636ceeea3637f'),
      ObjectId('6650a73d310636ceeea36380'),
      ObjectId('6650a73d310636ceeea36381'),
      ObjectId('6650a73d310636ceeea36382'),
      ObjectId('6650a73d310636ceeea36383'),
      ObjectId('6650a73d310636ceeea36384'),
      ObjectId('6650a73d310636ceeea36385'),
      ObjectId('6650a73d310636ceeea36386'),
      ObjectId('6650a73d310636ceeea36387'),
      ObjectId('6650a73d310636ceeea36388'),
      ObjectId('6650a73d310636ceeea36389'),
      ObjectId('6650a73d310636ceeea3638a'),
      ObjectId('6650a73d310636ceeea3638b'),
      ObjectId('6650a73d310636ceeea3638c'),
      ObjectId('6650a73d310636ceeea3638d'),
      ObjectId('6650a73d310636ceeea3638e'),
      ObjectId('6650a73d310636ceeea3638f'),
      ObjectId('6650a73d310636ceeea36390'),
      ObjectId('6650a73d310636ceeea36391'),
      ObjectId('6650a73d310636ceeea36392'),
      ObjectId('6650a73d310636ceeea36393'),
      ObjectId('6650a73d310636ceeea36394'),
      ObjectId('6650a73d310636ceeea36395'),
      ObjectId('6650a73d310636ceeea36396'),
      ObjectId('6650a73d310636ceeea36397'),
      ObjectId('6650a73d310636ceeea36398'),
      ObjectId('6650a73d310636ceeea36399'),
      ObjectId('6650a73d310636ceeea3639a'),
      ObjectId('6650a73d310636ceeea3639b'),
      ObjectId('6650a73d310636ceeea3639c'),
      ObjectId('6650a73d310636ceeea3639d'),
      ObjectId('6650a73d310636ceeea3639e'),
      ObjectId('6650a73d310636ceeea3639f'),
      ObjectId('6650a73d310636ceeea363a0'),
      ObjectId('6650a73d310636ceeea363a1'),
      ObjectId('6650a73d310636ceeea363a2'),
      ObjectId('6650a73d310636ceeea363a3'),
      ObjectId('6650a73d310636ceeea363a4'),
      ObjectId('6650a73d310636ceeea363a5'),
      ObjectId('6650a73d310636ceeea363a6'),
      ObjectId('6650a73d310636ceeea363a7'),
      ObjectId('6650a73d310636ceeea363a8'),
      ObjectId('6650a73d310636ceeea363a9'),
      ObjectId('6650a73d310636ceeea363aa'),
      ObjectId('6650a73d310636ceeea363ab'),
      ObjectId('6650a73d310636ceeea363ac'),
      ObjectId('6650a73d310636ceeea363ad'),
      ObjectId('6650a73d310636ceeea363ae'),
      ObjectId('6650a73d310636ceeea363af'),
      ObjectId('6650a73d310636ceeea363b0'),
      ObjectId('6650a73d310636ceeea363b1'),
      ObjectId('6650a73d310636ceeea363b2'),
      ObjectId('6650a73d310636ceeea363b3'),
      ObjectId('6650a73d310636ceeea363b4'),
      ObjectId('6650a73d310636ceeea363b5'),
      ObjectId('6650a73d310636ceeea363b6'),
      ObjectId('6650a73d310636ceeea363b7'),
      ObjectId('6650a73d310636ceeea363b8'),
      ObjectId('6650a73d310636ceeea363b9'),
      ObjectId('6650a73d310636ceeea363ba'),
      ObjectId('6650a73d310636ceeea363bb'),
      ObjectId('6650a73d310636ceeea363bc'),
      ObjectId('6650a73d310636ceeea363bd'),
      ObjectId('6650a73d310636ceeea363be'),
      ObjectId('6650a73d310636ceeea363bf'),
      ObjectId('6650a73d310636ceeea363c0'),
      ObjectId('6650a73d310636ceeea363c1'),
      ObjectId('6650a73d310636ceeea363c2'),
      ObjectId('6650a73d310636ceeea363c3'),
      ObjectId('6650a73d310636ceeea363c4'),
      ObjectId('6650a73d310636ceeea363c5'),
      ObjectId('6650a73d310636ceeea363c6'),
      ObjectId('6650a73d310636ceeea363c7'),
      ObjectId('6650a73d310636ceeea363c8'),
      ObjectId('6650a73d310636ceeea363c9'),
      ObjectId('6650a73d310636ceeea363ca'),
      ObjectId('6650a73d310636ceeea363cb'),
      ObjectId('6650a73d310636ceeea363cc'),
      ObjectId('6650a73d310636ceeea363cd'),
      ObjectId('6650a73d310636ceeea363ce'),
      ObjectId('6650a73d310636ceeea363cf'),
      ObjectId('6650a73d310636ceeea363d0'),
      ObjectId('6650a73d310636ceeea363d1'),
      ObjectId('6650a73d310636ceeea363d2'),
      ObjectId('6650a73d310636ceeea363d3'),
      ObjectId('6650a73d310636ceeea363d4'),
      ObjectId('6650a73d310636ceeea363d5'),
      ObjectId('6650a73d310636ceeea363d6'),
      ObjectId('6650a73d310636ceeea363d7'),
      ObjectId('6650a73d310636ceeea363d8'),
      ObjectId('6650a73d310636ceeea363d9'),
      ObjectId('6650a73d310636ceeea363da'),
      ObjectId('6650a73d310636ceeea363db'),
      ObjectId('6650a73d310636ceeea363dc'),
      ObjectId('6650a73d310636ceeea363dd'),
      ObjectId('6650a73d310636ceeea363de'),
      ObjectId('6650a73d310636ceeea363df'),
      ObjectId('6650a73d310636ceeea363e0'),
      ObjectId('6650a73d310636ceeea363e1'),
      ObjectId('6650a73d310636ceeea363e2'),
      ObjectId('6650a73d310636ceeea363e3'),
      ObjectId('6650a73d310636ceeea363e4'),
      ObjectId('6650a73d310636ceeea363e5'),
      ObjectId('6650a73d310636ceeea363e6'),
      ObjectId('6650a73d310636ceeea363e7'),
      ObjectId('6650a73d310636ceeea363e8'),
      ObjectId('6650a73d310636ceeea363e9'),
      ObjectId('6650a73d310636ceeea363ea'),
      ObjectId('6650a73d310636ceeea363eb'),
      ObjectId('6650a73d310636ceeea363ec'),
      ObjectId('6650a73d310636ceeea363ed'),
      ObjectId('6650a73d310636ceeea363ee'),
      ObjectId('6650a73d310636ceeea363ef'),
      ObjectId('6650a73d310636ceeea363f0'),
      ObjectId('6650a73d310636ceeea363f1'),
      ObjectId('6650a73d310636ceeea363f2'),
      ObjectId('6650a73d310636ceeea363f3'),
      ObjectId('6650a73d310636ceeea363f4'),
      ObjectId('6650a73d310636ceeea363f5'),
      ObjectId('6650a73d310636ceeea363f6'),
      ObjectId('6650a73d310636ceeea363f7'),
      ObjectId('6650a73d310636ceeea363f8'),
      ObjectId('6650a73d310636ceeea363f9'),
      ObjectId('6650a73d310636ceeea363fa'),
      ObjectId('6650a73d310636ceeea363fb'),
      ObjectId('6650a73d310636ceeea363fc'),
      ObjectId('6650a73d310636ceeea363fd'),
      ObjectId('6650a73d310636ceeea363fe'),
      ObjectId('6650a73d310636ceeea363ff'),
      ObjectId('6650a73d310636ceeea36400'),
      ObjectId('6650a73d310636ceeea36401'),
      ObjectId('6650a73d310636ceeea36402'),
      ObjectId('6650a73d310636ceeea36403'),
      ObjectId('6650a73d310636ceeea36404'),
      ObjectId('6650a73d310636ceeea36405'),
      ObjectId('6650a73d310636ceeea36406'),
      ObjectId('6650a73d310636ceeea36407'),
      ObjectId('6650a73d310636ceeea36408'),
      ObjectId('6650a73d310636ceeea36409'),
      ObjectId('6650a73d310636ceeea3640a'),
      ObjectId('6650a73d310636ceeea3640b'),
      ObjectId('6650a73d310636ceeea3640c'),
      ObjectId('6650a73d310636ceeea3640d'),
      ObjectId('6650a73d310636ceeea3640e'),
      ObjectId('6650a73d310636ceeea3640f'),
      ObjectId('6650a73d310636ceeea36410'),
      ObjectId('6650a73d310636ceeea36411'),
      ObjectId('6650a73d310636ceeea36412'),
      ObjectId('6650a73d310636ceeea36413'),
      ObjectId('6650a73d310636ceeea36414'),
      ObjectId('6650a73d310636ceeea36415'),
      ObjectId('6650a73d310636ceeea36416'),
      ObjectId('6650a73d310636ceeea36417'),
      ObjectId('6650a73d310636ceeea36418'),
      ObjectId('6650a73d310636ceeea36419'),
      ObjectId('6650a73d310636ceeea3641a'),
      ObjectId('6650a73d310636ceeea3641b'),
      ObjectId('6650a73d310636ceeea3641c'),
      ObjectId('6650a73d310636ceeea3641d'),
      ObjectId('6650a73d310636ceeea3641e'),
      ObjectId('6650a73d310636ceeea3641f'),
      ObjectId('6650a73d310636ceeea36420'),
      ObjectId('6650a73d310636ceeea36421'),
      ObjectId('6650a73d310636ceeea36422'),
      ObjectId('6650a73d310636ceeea36423'),
      ObjectId('6650a73d310636ceeea36424'),
      ObjectId('6650a73d310636ceeea36425'),
      ObjectId('6650a73d310636ceeea36426'),
      ObjectId('6650a73d310636ceeea36427'),
      ObjectId('6650a73d310636ceeea36428'),
      ObjectId('6650a73d310636ceeea36429'),
      ObjectId('6650a73d310636ceeea3642a'),
      ObjectId('6650a73d310636ceeea3642b'),
      ObjectId('6650a73d310636ceeea3642c'),
      ObjectId('6650a73d310636ceeea3642d'),
      ObjectId('6650a73d310636ceeea3642e'),
      ObjectId('6650a73d310636ceeea3642f'),
      ObjectId('6650a73d310636ceeea36430'),
      ObjectId('6650a73d310636ceeea36431'),
      ObjectId('6650a73d310636ceeea36432'),
      ObjectId('6650a73d310636ceeea36433'),
      ObjectId('6650a73d310636ceeea36434'),
      ObjectId('6650a73d310636ceeea36435'),
      ObjectId('6650a73d310636ceeea36436'),
      ObjectId('6650a73d310636ceeea36437'),
      ObjectId('6650a73d310636ceeea36438'),
      ObjectId('6650a73d310636ceeea36439'),
      ObjectId('6650a73d310636ceeea3643a'),
      ObjectId('6650a73d310636ceeea3643b'),
      ObjectId('6650a73d310636ceeea3643c'),
      ObjectId('6650a73d310636ceeea3643d'),
      ObjectId('6650a73d310636ceeea3643e'),
      ObjectId('6650a73d310636ceeea3643f'),
      ObjectId('6650a73d310636ceeea36440'),
      ObjectId('6650a73d310636ceeea36441'),
      ObjectId('6650a73d310636ceeea36442'),
      ObjectId('6650a73d310636ceeea36443'),
      ObjectId('6650a73d310636ceeea36444'),
      ObjectId('6650a73d310636ceeea36445'),
      ObjectId('6650a73d310636ceeea36446'),
      ObjectId('6650a73d310636ceeea36447'),
      ObjectId('6650a73d310636ceeea36448'),
      ObjectId('6650a73d310636ceeea36449'),
      ObjectId('6650a73d310636ceeea3644a'),
      ObjectId('6650a73d310636ceeea3644b'),
      ObjectId('6650a73d310636ceeea3644c'),
      ObjectId('6650a73d310636ceeea3644d'),
      ObjectId('6650a73d310636ceeea3644e'),
      ObjectId('6650a73d310636ceeea3644f'),
      ObjectId('6650a73d310636ceeea36450'),
      ObjectId('6650a73d310636ceeea36451'),
      ObjectId('6650a73d310636ceeea36452'),
      ObjectId('6650a73d310636ceeea36453'),
      ObjectId('6650a73d310636ceeea36454'),
      ObjectId('6650a73d310636ceeea36455'),
      ObjectId('6650a73d310636ceeea36456'),
      ObjectId('6650a73d310636ceeea36457'),
      ObjectId('6650a73d310636ceeea36458'),
      ObjectId('6650a73d310636ceeea36459'),
      ObjectId('6650a73d310636ceeea3645a'),
      ObjectId('6650a73d310636ceeea3645b'),
      ObjectId('6650a73d310636ceeea3645c'),
      ObjectId('6650a73d310636ceeea3645d'),
      ObjectId('6650a73d310636ceeea3645e'),
      ObjectId('6650a73d310636ceeea3645f'),
      ObjectId('6650a73d310636ceeea36460'),
      ObjectId('6650a73d310636ceeea36461'),
      ObjectId('6650a73d310636ceeea36462'),
      ObjectId('6650a73d310636ceeea36463'),
      ObjectId('6650a73d310636ceeea36464'),
      ObjectId('6650a73d310636ceeea36465'),
      ObjectId('6650a73d310636ceeea36466'),
      ObjectId('6650a73d310636ceeea36467'),
      ObjectId('6650a73d310636ceeea36468'),
      ObjectId('6650a73d310636ceeea36469'),
      ObjectId('6650a73d310636ceeea3646a'),
      ObjectId('6650a73d310636ceeea3646b'),
      ObjectId('6650a73d310636ceeea3646c'),
      ObjectId('6650a73d310636ceeea3646d'),
      ObjectId('6650a73d310636ceeea3646e'),
      ObjectId('6650a73d310636ceeea3646f'),
      ObjectId('6650a73d310636ceeea36470'),
      ObjectId('6650a73d310636ceeea36471'),
      ObjectId('6650a73d310636ceeea36472'),
      ObjectId('6650a73d310636ceeea36473'),
      ObjectId('6650a73d310636ceeea36474'),
      ObjectId('6650a73d310636ceeea36475'),
      ObjectId('6650a73d310636ceeea36476'),
      ObjectId('6650a73d310636ceeea36477'),
      ObjectId('6650a73d310636ceeea36478'),
      ObjectId('6650a73d310636ceeea36479'),
      ObjectId('6650a73d310636ceeea3647a'),
      ObjectId('6650a73d310636ceeea3647b'),
      ObjectId('6650a73d310636ceeea3647c'),
      ObjectId('6650a73d310636ceeea3647d'),
      ObjectId('6650a73d310636ceeea3647e'),
      ObjectId('6650a73d310636ceeea3647f'),
      ObjectId('6650a73d310636ceeea36480'),
      ObjectId('6650a73d310636ceeea36481'),
      ObjectId('6650a73d310636ceeea36482'),
      ObjectId('6650a73d310636ceeea36483'),
      ObjectId('6650a73d310636ceeea36484'),
      ObjectId('6650a73d310636ceeea36485'),
      ObjectId('6650a73d310636ceeea36486'),
      ObjectId('6650a73d310636ceeea36487'),
      ObjectId('6650a73d310636ceeea36488'),
      ObjectId('6650a73d310636ceeea36489'),
      ObjectId('6650a73d310636ceeea3648a'),
      ObjectId('6650a73d310636ceeea3648b'),
      ObjectId('6650a73d310636ceeea3648c'),
      ObjectId('6650a73d310636ceeea3648d'),
      ObjectId('6650a73d310636ceeea3648e'),
      ObjectId('6650a73d310636ceeea3648f'),
      ObjectId('6650a73d310636ceeea36490'),
      ObjectId('6650a73d310636ceeea36491'),
      ObjectId('6650a73d310636ceeea36492'),
      ObjectId('6650a73d310636ceeea36493'),
      ObjectId('6650a73d310636ceeea36494'),
      ObjectId('6650a73d310636ceeea36495'),
      ObjectId('6650a73d310636ceeea36496'),
      ObjectId('6650a73d310636ceeea36497'),
      ObjectId('6650a73d310636ceeea36498'),
      ObjectId('6650a73d310636ceeea36499'),
      ObjectId('6650a73d310636ceeea3649a'),
      ObjectId('6650a73d310636ceeea3649b'),
      ObjectId('6650a73d310636ceeea3649c'),
      ObjectId('6650a73d310636ceeea3649d'),
      ObjectId('6650a73d310636ceeea3649e'),
      ObjectId('6650a73d310636ceeea3649f'),
      ObjectId('6650a73d310636ceeea364a0'),
      ObjectId('6650a73d310636ceeea364a1'),
      ObjectId('6650a73d310636ceeea364a2'),
      ObjectId('6650a73d310636ceeea364a3'),
      ObjectId('6650a73d310636ceeea364a4'),
      ObjectId('6650a73d310636ceeea364a5'),
      ObjectId('6650a73d310636ceeea364a6'),
      ObjectId('6650a73d310636ceeea364a7'),
      ObjectId('6650a73d310636ceeea364a8'),
      ObjectId('6650a73d310636ceeea364a9'),
      ObjectId('6650a73d310636ceeea364aa'),
      ObjectId('6650a73d310636ceeea364ab'),
      ObjectId('6650a73d310636ceeea364ac'),
      ObjectId('6650a73d310636ceeea364ad'),
      ObjectId('6650a73d310636ceeea364ae'),
      ObjectId('6650a73d310636ceeea364af'),
      ObjectId('6650a73d310636ceeea364b0'),
      ObjectId('6650a73d310636ceeea364b1'),
      ObjectId('6650a73d310636ceeea364b2'),
      ObjectId('6650a73d310636ceeea364b3'),
      ObjectId('6650a73d310636ceeea364b4'),
      ObjectId('6650a73d310636ceeea364b5'),
      ObjectId('6650a73d310636ceeea364b6'),
      ObjectId('6650a73d310636ceeea364b7'),
      ObjectId('6650a73d310636ceeea364b8'),
      ObjectId('6650a73d310636ceeea364b9'),
      ObjectId('6650a73d310636ceeea364ba'),
      ObjectId('6650a73d310636ceeea364bb'),
      ObjectId('6650a73d310636ceeea364bc'),
      ObjectId('6650a73d310636ceeea364bd'),
      ObjectId('6650a73d310636ceeea364be'),
      ObjectId('6650a73d310636ceeea364bf'),
      ObjectId('6650a73d310636ceeea364c0'),
      ObjectId('6650a73d310636ceeea364c1'),
      ObjectId('6650a73d310636ceeea364c2'),
      ObjectId('6650a73d310636ceeea364c3'),
      ObjectId('6650a73d310636ceeea364c4'),
      ObjectId('6650a73d310636ceeea364c5'),
      ObjectId('6650a73d310636ceeea364c6'),
      ObjectId('6650a73d310636ceeea364c7'),
      ObjectId('6650a73d310636ceeea364c8'),
      ObjectId('6650a73d310636ceeea364c9'),
      ObjectId('6650a73d310636ceeea364ca'),
      ObjectId('6650a73d310636ceeea364cb'),
      ObjectId('6650a73d310636ceeea364cc'),
      ObjectId('6650a73d310636ceeea364cd'),
      ObjectId('6650a73d310636ceeea364ce'),
      ObjectId('6650a73d310636ceeea364cf'),
      ObjectId('6650a73d310636ceeea364d0'),
      ObjectId('6650a73d310636ceeea364d1'),
      ObjectId('6650a73d310636ceeea364d2'),
      ObjectId('6650a73d310636ceeea364d3'),
      ObjectId('6650a73d310636ceeea364d4'),
      ObjectId('6650a73d310636ceeea364d5'),
      ObjectId('6650a73d310636ceeea364d6'),
      ObjectId('6650a73d310636ceeea364d7'),
      ObjectId('6650a73d310636ceeea364d8'),
      ObjectId('6650a73d310636ceeea364d9'),
      ObjectId('6650a73d310636ceeea364da'),
      ObjectId('6650a73d310636ceeea364db'),
      ObjectId('6650a73d310636ceeea364dc'),
      ObjectId('6650a73d310636ceeea364dd'),
      ObjectId('6650a73d310636ceeea364de'),
      ObjectId('6650a73d310636ceeea364df'),
      ObjectId('6650a73d310636ceeea364e0'),
      ObjectId('6650a73d310636ceeea364e1'),
      ObjectId('6650a73d310636ceeea364e2'),
      ObjectId('6650a73d310636ceeea364e3'),
      ObjectId('6650a73d310636ceeea364e4'),
      ObjectId('6650a73d310636ceeea364e5'),
      ObjectId('6650a73d310636ceeea364e6'),
      ObjectId('6650a73d310636ceeea364e7'),
      ObjectId('6650a73d310636ceeea364e8'),
      ObjectId('6650a73d310636ceeea364e9'),
      ObjectId('6650a73d310636ceeea364ea'),
      ObjectId('6650a73d310636ceeea364eb'),
      ObjectId('6650a73d310636ceeea364ec'),
      ObjectId('6650a73d310636ceeea364ed'),
      ObjectId('6650a73d310636ceeea364ee'),
      ObjectId('6650a73d310636ceeea364ef'),
      ObjectId('6650a73d310636ceeea364f0'),
      ObjectId('6650a73d310636ceeea364f1'),
      ObjectId('6650a73d310636ceeea364f2'),
      ObjectId('6650a73d310636ceeea364f3'),
      ObjectId('6650a73d310636ceeea364f4'),
      ObjectId('6650a73d310636ceeea364f5'),
      ObjectId('6650a73d310636ceeea364f6'),
      ObjectId('6650a73d310636ceeea364f7'),
      ObjectId('6650a73d310636ceeea364f8'),
      ObjectId('6650a73d310636ceeea364f9'),
      ObjectId('6650a73d310636ceeea364fa'),
      ObjectId('6650a73d310636ceeea364fb'),
      ObjectId('6650a73d310636ceeea364fc'),
      ObjectId('6650a73d310636ceeea364fd'),
      ObjectId('6650a73d310636ceeea364fe'),
      ObjectId('6650a73d310636ceeea364ff'),
      ObjectId('6650a73d310636ceeea36500'),
      ObjectId('6650a73d310636ceeea36501'),
      ObjectId('6650a73d310636ceeea36502'),
      ObjectId('6650a73d310636ceeea36503'),
      ObjectId('6650a73d310636ceeea36504'),
      ObjectId('6650a73d310636ceeea36505'),
      ObjectId('6650a73d310636ceeea36506'),
      ObjectId('6650a73d310636ceeea36507'),
      ObjectId('6650a73d310636ceeea36508'),
      ObjectId('6650a73d310636ceeea36509'),
      ObjectId('6650a73d310636ceeea3650a'),
      ObjectId('6650a73d310636ceeea3650b'),
      ObjectId('6650a73d310636ceeea3650c'),
      ObjectId('6650a73d310636ceeea3650d'),
      ObjectId('6650a73d310636ceeea3650e'),
      ObjectId('6650a73d310636ceeea3650f'),
      ObjectId('6650a73d310636ceeea36510'),
      ObjectId('6650a73d310636ceeea36511'),
      ObjectId('6650a73d310636ceeea36512'),
      ObjectId('6650a73d310636ceeea36513'),
      ObjectId('6650a73d310636ceeea36514'),
      ObjectId('6650a73d310636ceeea36515'),
      ObjectId('6650a73d310636ceeea36516'),
      ObjectId('6650a73d310636ceeea36517'),
      ObjectId('6650a73d310636ceeea36518'),
      ObjectId('6650a73d310636ceeea36519'),
      ObjectId('6650a73d310636ceeea3651a'),
      ObjectId('6650a73d310636ceeea3651b'),
      ObjectId('6650a73d310636ceeea3651c'),
      ObjectId('6650a73d310636ceeea3651d'),
      ObjectId('6650a73d310636ceeea3651e'),
      ObjectId('6650a73d310636ceeea3651f'),
      ObjectId('6650a73d310636ceeea36520'),
      ObjectId('6650a73d310636ceeea36521'),
      ObjectId('6650a73d310636ceeea36522'),
      ObjectId('6650a73d310636ceeea36523'),
      ObjectId('6650a73d310636ceeea36524'),
      ObjectId('6650a73d310636ceeea36525'),
      ObjectId('6650a73d310636ceeea36526'),
      ObjectId('6650a73d310636ceeea36527'),
      ObjectId('6650a73d310636ceeea36528'),
      ObjectId('6650a73d310636ceeea36529'),
      ObjectId('6650a73d310636ceeea3652a'),
      ObjectId('6650a73d310636ceeea3652b'),
      ObjectId('6650a73d310636ceeea3652c'),
      ObjectId('6650a73d310636ceeea3652d'),
      ObjectId('6650a73d310636ceeea3652e'),
      ObjectId('6650a73d310636ceeea3652f'),
      ObjectId('6650a73d310636ceeea36530'),
      ObjectId('6650a73d310636ceeea36531'),
      ObjectId('6650a73d310636ceeea36532'),
      ObjectId('6650a73d310636ceeea36533'),
      ObjectId('6650a73d310636ceeea36534'),
      ObjectId('6650a73d310636ceeea36535'),
      ObjectId('6650a73d310636ceeea36536'),
      ObjectId('6650a73d310636ceeea36537'),
      ObjectId('6650a73d310636ceeea36538'),
      ObjectId('6650a73d310636ceeea36539'),
      ObjectId('6650a73d310636ceeea3653a'),
      ObjectId('6650a73d310636ceeea3653b'),
      ObjectId('6650a73d310636ceeea3653c'),
      ObjectId('6650a73d310636ceeea3653d'),
      ObjectId('6650a73d310636ceeea3653e'),
      ObjectId('6650a73d310636ceeea3653f'),
      ObjectId('6650a73d310636ceeea36540'),
      ObjectId('6650a73d310636ceeea36541'),
      ObjectId('6650a73d310636ceeea36542'),
      ObjectId('6650a73d310636ceeea36543'),
      ObjectId('6650a73d310636ceeea36544'),
      ObjectId('6650a73d310636ceeea36545'),
      ObjectId('6650a73d310636ceeea36546'),
      ObjectId('6650a73d310636ceeea36547'),
      ObjectId('6650a73d310636ceeea36548'),
      ObjectId('6650a73d310636ceeea36549'),
      ObjectId('6650a73d310636ceeea3654a'),
      ObjectId('6650a73d310636ceeea3654b'),
      ObjectId('6650a73d310636ceeea3654c'),
      ObjectId('6650a73d310636ceeea3654d'),
      ObjectId('6650a73d310636ceeea3654e'),
      ObjectId('6650a73d310636ceeea3654f'),
      ObjectId('6650a73d310636ceeea36550'),
      ObjectId('6650a73d310636ceeea36551'),
      ObjectId('6650a73d310636ceeea36552'),
      ObjectId('6650a73d310636ceeea36553'),
      ObjectId('6650a73d310636ceeea36554'),
      ObjectId('6650a73d310636ceeea36555'),
      ObjectId('6650a73d310636ceeea36556'),
      ObjectId('6650a73d310636ceeea36557'),
      ObjectId('6650a73d310636ceeea36558'),
      ObjectId('6650a73d310636ceeea36559'),
      ObjectId('6650a73d310636ceeea3655a'),
      ObjectId('6650a73d310636ceeea3655b'),
      ObjectId('6650a73d310636ceeea3655c'),
      ObjectId('6650a73d310636ceeea3655d'),
      ObjectId('6650a73d310636ceeea3655e'),
      ObjectId('6650a73d310636ceeea3655f'),
      ObjectId('6650a73d310636ceeea36560'),
      ObjectId('6650a73d310636ceeea36561'),
      ObjectId('6650a73d310636ceeea36562'),
      ObjectId('6650a73d310636ceeea36563'),
      ObjectId('6650a73d310636ceeea36564'),
      ObjectId('6650a73d310636ceeea36565'),
      ObjectId('6650a73d310636ceeea36566'),
      ObjectId('6650a73d310636ceeea36567'),
      ObjectId('6650a73d310636ceeea36568'),
      ObjectId('6650a73d310636ceeea36569'),
      ObjectId('6650a73d310636ceeea3656a'),
      ObjectId('6650a73d310636ceeea3656b'),
      ObjectId('6650a73d310636ceeea3656c'),
      ObjectId('6650a73d310636ceeea3656d'),
      ObjectId('6650a73d310636ceeea3656e'),
      ObjectId('6650a73d310636ceeea3656f'),
      ObjectId('6650a73d310636ceeea36570'),
      ObjectId('6650a73d310636ceeea36571'),
      ObjectId('6650a73d310636ceeea36572'),
      ObjectId('6650a73d310636ceeea36573'),
      ObjectId('6650a73d310636ceeea36574'),
      ObjectId('6650a73d310636ceeea36575'),
      ObjectId('6650a73d310636ceeea36576'),
      ObjectId('6650a73d310636ceeea36577'),
      ObjectId('6650a73d310636ceeea36578'),
      ObjectId('6650a73d310636ceeea36579'),
      ObjectId('6650a73d310636ceeea3657a'),
      ObjectId('6650a73d310636ceeea3657b'),
      ObjectId('6650a73d310636ceeea3657c'),
      ObjectId('6650a73d310636ceeea3657d'),
      ObjectId('6650a73d310636ceeea3657e'),
      ObjectId('6650a73d310636ceeea3657f'),
      ObjectId('6650a73d310636ceeea36580'),
      ObjectId('6650a73d310636ceeea36581'),
      ObjectId('6650a73d310636ceeea36582'),
      ObjectId('6650a73d310636ceeea36583'),
      ObjectId('6650a73d310636ceeea36584'),
      ObjectId('6650a73d310636ceeea36585'),
      ObjectId('6650a73d310636ceeea36586'),
      ObjectId('6650a73d310636ceeea36587'),
      ObjectId('6650a73d310636ceeea36588'),
      ObjectId('6650a73d310636ceeea36589'),
      ObjectId('6650a73d310636ceeea3658a'),
      ObjectId('6650a73d310636ceeea3658b'),
      ObjectId('6650a73d310636ceeea3658c'),
      ObjectId('6650a73d310636ceeea3658d'),
      ObjectId('6650a73d310636ceeea3658e'),
      ObjectId('6650a73d310636ceeea3658f'),
      ObjectId('6650a73d310636ceeea36590'),
      ObjectId('6650a73d310636ceeea36591'),
      ObjectId('6650a73d310636ceeea36592'),
      ObjectId('6650a73d310636ceeea36593'),
      ObjectId('6650a73d310636ceeea36594'),
      ObjectId('6650a73d310636ceeea36595'),
      ObjectId('6650a73d310636ceeea36596'),
      ObjectId('6650a73d310636ceeea36597'),
      ObjectId('6650a73d310636ceeea36598'),
      ObjectId('6650a73d310636ceeea36599'),
      ObjectId('6650a73d310636ceeea3659a'),
      ObjectId('6650a73d310636ceeea3659b'),
      ObjectId('6650a73d310636ceeea3659c'),
      ObjectId('6650a73d310636ceeea3659d'),
      ObjectId('6650a73d310636ceeea3659e'),
      ObjectId('6650a73d310636ceeea3659f'),
      ObjectId('6650a73d310636ceeea365a0'),
      ObjectId('6650a73d310636ceeea365a1'),
      ObjectId('6650a73d310636ceeea365a2'),
      ObjectId('6650a73d310636ceeea365a3'),
      ObjectId('6650a73d310636ceeea365a4'),
      ObjectId('6650a73d310636ceeea365a5'),
      ObjectId('6650a73d310636ceeea365a6'),
      ObjectId('6650a73d310636ceeea365a7'),
      ObjectId('6650a73d310636ceeea365a8'),
      ObjectId('6650a73d310636ceeea365a9'),
      ObjectId('6650a73d310636ceeea365aa'),
      ObjectId('6650a73d310636ceeea365ab'),
      ObjectId('6650a73d310636ceeea365ac'),
      ObjectId('6650a73d310636ceeea365ad'),
      ObjectId('6650a73d310636ceeea365ae'),
      ObjectId('6650a73d310636ceeea365af'),
      ObjectId('6650a73d310636ceeea365b0'),
      ObjectId('6650a73d310636ceeea365b1'),
      ObjectId('6650a73d310636ceeea365b2'),
      ObjectId('6650a73d310636ceeea365b3'),
      ObjectId('6650a73d310636ceeea365b4'),
      ObjectId('6650a73d310636ceeea365b5'),
      ObjectId('6650a73d310636ceeea365b6'),
      ObjectId('6650a73d310636ceeea365b7'),
      ObjectId('6650a73d310636ceeea365b8'),
      ObjectId('6650a73d310636ceeea365b9'),
      ObjectId('6650a73d310636ceeea365ba'),
      ObjectId('6650a73d310636ceeea365bb'),
      ObjectId('6650a73d310636ceeea365bc'),
      ObjectId('6650a73d310636ceeea365bd'),
      ObjectId('6650a73d310636ceeea365be'),
      ObjectId('6650a73d310636ceeea365bf'),
      ObjectId('6650a73d310636ceeea365c0'),
      ObjectId('6650a73d310636ceeea365c1'),
      ObjectId('6650a73d310636ceeea365c2'),
      ObjectId('6650a73d310636ceeea365c3'),
      ObjectId('6650a73d310636ceeea365c4'),
      ObjectId('6650a73d310636ceeea365c5'),
      ObjectId('6650a73d310636ceeea365c6'),
      ObjectId('6650a73d310636ceeea365c7'),
      ObjectId('6650a73d310636ceeea365c8'),
      ObjectId('6650a73d310636ceeea365c9'),
      ObjectId('6650a73d310636ceeea365ca'),
      ObjectId('6650a73d310636ceeea365cb'),
      ObjectId('6650a73d310636ceeea365cc'),
      ObjectId('6650a73d310636ceeea365cd'),
      ObjectId('6650a73d310636ceeea365ce'),
      ObjectId('6650a73d310636ceeea365cf'),
      ObjectId('6650a73d310636ceeea365d0'),
      ObjectId('6650a73d310636ceeea365d1'),
      ObjectId('6650a73d310636ceeea365d2'),
      ObjectId('6650a73d310636ceeea365d3'),
      ObjectId('6650a73d310636ceeea365d4'),
      ObjectId('6650a73d310636ceeea365d5'),
      ObjectId('6650a73d310636ceeea365d6'),
      ObjectId('6650a73d310636ceeea365d7'),
      ObjectId('6650a73d310636ceeea365d8'),
      ObjectId('6650a73d310636ceeea365d9'),
      ObjectId('6650a73d310636ceeea365da'),
      ObjectId('6650a73d310636ceeea365db'),
      ObjectId('6650a73d310636ceeea365dc'),
      ObjectId('6650a73d310636ceeea365dd'),
      ObjectId('6650a73d310636ceeea365de'),
      ObjectId('6650a73d310636ceeea365df'),
      ObjectId('6650a73d310636ceeea365e0'),
      ObjectId('6650a73d310636ceeea365e1'),
      ObjectId('6650a73d310636ceeea365e2'),
      ObjectId('6650a73d310636ceeea365e3'),
      ObjectId('6650a73d310636ceeea365e4'),
      ObjectId('6650a73d310636ceeea365e5'),
      ObjectId('6650a73d310636ceeea365e6'),
      ObjectId('6650a73d310636ceeea365e7'),
      ObjectId('6650a73d310636ceeea365e8'),
      ObjectId('6650a73d310636ceeea365e9'),
      ObjectId('6650a73d310636ceeea365ea'),
      ObjectId('6650a73d310636ceeea365eb'),
      ObjectId('6650a73d310636ceeea365ec'),
      ObjectId('6650a73d310636ceeea365ed'),
      ObjectId('6650a73d310636ceeea365ee'),
      ObjectId('6650a73d310636ceeea365ef'),
      ObjectId('6650a73d310636ceeea365f0'),
      ObjectId('6650a73d310636ceeea365f1'),
      ObjectId('6650a73d310636ceeea365f2'),
      ObjectId('6650a73d310636ceeea365f3'),
      ObjectId('6650a73d310636ceeea365f4'),
      ObjectId('6650a73d310636ceeea365f5'),
      ObjectId('6650a73d310636ceeea365f6'),
      ObjectId('6650a73d310636ceeea365f7'),
      ObjectId('6650a73d310636ceeea365f8'),
      ObjectId('6650a73d310636ceeea365f9'),
      ObjectId('6650a73d310636ceeea365fa'),
      ObjectId('6650a73d310636ceeea365fb'),
      ObjectId('6650a73d310636ceeea365fc'),
      ObjectId('6650a73d310636ceeea365fd'),
      ObjectId('6650a73d310636ceeea365fe'),
      ObjectId('6650a73d310636ceeea365ff'),
      ObjectId('6650a73d310636ceeea36600'),
      ObjectId('6650a73d310636ceeea36601'),
      ObjectId('6650a73d310636ceeea36602'),
      ObjectId('6650a73d310636ceeea36603'),
      ObjectId('6650a73d310636ceeea36604'),
      ObjectId('6650a73d310636ceeea36605'),
      ObjectId('6650a73d310636ceeea36606'),
      ObjectId('6650a73d310636ceeea36607'),
      ObjectId('6650a73d310636ceeea36608'),
      ObjectId('6650a73d310636ceeea36609'),
      ObjectId('6650a73d310636ceeea3660a'),
      ObjectId('6650a73d310636ceeea3660b'),
      ObjectId('6650a73d310636ceeea3660c'),
      ObjectId('6650a73d310636ceeea3660d'),
      ObjectId('6650a73d310636ceeea3660e'),
      ObjectId('6650a73d310636ceeea3660f'),
      ObjectId('6650a73d310636ceeea36610')],
     TaskWorkflow(database=\<superduperdb.base.datalayer.Datalayer object at 0x1520664d0\>, G=\<networkx.classes.digraph.DiGraph object at 0x155c58350\>))
</pre>
</details>

Now that the images and their classes are inserted into the database, we can query the data in its original format. Particularly, we can use the `PIL.Image` instances to inspect the data.

```python
# Get and display one of the images
r = db['mnist'].find_one().execute()
r.unpack()['img'].resize((300, 300))
```

<details>
<summary>Outputs</summary>
<div>![](/training/8_0.png)</div>
</details>

Following that, we build our machine learning model. SuperDuperDB conveniently supports various frameworks, and for this example, we opt for PyTorch, a suitable choice for computer vision tasks. In this instance, we combine `torch` with `torchvision`.

To facilitate communication with the SuperDuperDB `Datalayer`, we design `postprocess` and `preprocess` functions. These functions are then wrapped with the `TorchModel` wrapper to create a native SuperDuperDB object.

```python
from superduperdb.ext.torch import TorchModel

import torch

# Define the LeNet-5 architecture for image classification
class LeNet5(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Layer 1
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # Layer 2
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # Fully connected layers
        self.fc = torch.nn.Linear(400, 120)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(120, 84)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# Postprocess function for the model output    
def postprocess(x):
    return int(x.topk(1)[1].item())

# Preprocess function for input data
def preprocess(x):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
    )(x)

# Create an instance of the LeNet-5 model
lenet_model = LeNet5(10)


model = TorchModel(
    identifier='my-model',
    object=lenet_model,
    preprocess=preprocess,
    postprocess=postprocess, 
    preferred_devices=('cpu',),
)

# Check that the model successfully creates predictions over single data-points
model.predict_one(data[0]['img'])
```

<details>
<summary>Outputs</summary>
<pre>
    2024-May-24 16:42:15.26| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.components.component:386  | Initializing TorchModel : my-model
    2024-May-24 16:42:15.27| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.components.component:389  | Initialized  TorchModel : my-model successfully

</pre>
<pre>
    2
</pre>
</details>

Now we are ready to "train" or "fit" the model. Trainable models in SuperDuperDB come with a sklearn-like `.fit` method,
which developers may implement for their specific model class. `torch` models come with a pre-configured
`TorchTrainer` class and `.fit` method. These may be invoked simply by "applying" the model to `db`:

```python
from torch.nn.functional import cross_entropy

from superduperdb import Metric, Validation, Dataset
from superduperdb.ext.torch import TorchTrainer

acc = lambda x, y: (sum([xx == yy for xx, yy in zip(x, y)]) / len(x))

accuracy = Metric(identifier='acc', object=acc)

model.validation = Validation(
    'mnist_performance',
    datasets=[
        Dataset(
            identifier='my-valid',
            select=db['mnist'].find({'_fold': 'valid'})
        )
    ],
    metrics=[accuracy],
)

model.trainer = TorchTrainer(
    identifier='my-trainer',
    objective=cross_entropy,
    loader_kwargs={'batch_size': 10},
    max_iterations=1000,
    validation_interval=5,
    select=db['mnist'].find(),
    key=('img', 'class'),
    transform=lambda x, y: (preprocess(x), y),
)

_ = db.apply(model)
```

<details>
<summary>Outputs</summary>
<pre>
    2024-May-24 16:42:19.76| WARNING  | Duncans-MacBook-Pro.fritz.box| superduperdb.backends.local.artifacts:82   | File /tmp/a10fbf2cdd7532dd7bf5bba03b7c28e01b4326cc already exists
    2024-May-24 16:42:19.79| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function method_job at 0x110261d00\>
    2024-May-24 16:42:19.92| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 0; objective: 2.30452561378479; 
    2024-May-24 16:42:19.94| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.components.component:386  | Initializing Dataset : my-valid
    2024-May-24 16:42:19.94| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.components.component:389  | Initialized  Dataset : my-valid successfully
    2024-May-24 16:42:19.94| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.components.component:386  | Initializing TorchModel : my-model
    2024-May-24 16:42:19.94| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.components.component:389  | Initialized  TorchModel : my-model successfully

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 963.46it/s]

</pre>
<pre>
    2024-May-24 16:42:19.99| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 0; my-valid/acc: 0.16279069767441862; objective: 2.3071595191955567; 
    2024-May-24 16:42:20.01| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 1; objective: 2.290095806121826; 
    2024-May-24 16:42:20.02| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 2; objective: 2.223555088043213; 
    2024-May-24 16:42:20.02| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 3; objective: 2.3189988136291504; 
    2024-May-24 16:42:20.03| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 4; objective: 2.1752736568450928; 
    2024-May-24 16:42:20.03| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 5; objective: 2.207839250564575; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 945.72it/s]

</pre>
<pre>
    2024-May-24 16:42:20.09| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 5; my-valid/acc: 0.11627906976744186; objective: 2.2843324184417724; 
    2024-May-24 16:42:20.11| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 6; objective: 2.282257318496704; 
    2024-May-24 16:42:20.12| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 7; objective: 2.0524115562438965; 
    2024-May-24 16:42:20.13| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 8; objective: 2.0606937408447266; 
    2024-May-24 16:42:20.14| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 9; objective: 2.194944143295288; 
    2024-May-24 16:42:20.14| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 10; objective: 2.29692006111145; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 772.76it/s]

</pre>
<pre>
    2024-May-24 16:42:20.22| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 10; my-valid/acc: 0.16279069767441862; objective: 2.2262439727783203; 
    2024-May-24 16:42:20.25| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 11; objective: 2.1308627128601074; 
    2024-May-24 16:42:20.26| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 12; objective: 2.155353307723999; 
    2024-May-24 16:42:20.27| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 13; objective: 2.0958755016326904; 
    2024-May-24 16:42:20.28| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 14; objective: 1.9480855464935303; 
    2024-May-24 16:42:20.28| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 15; objective: 1.9860684871673584; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 718.64it/s]

</pre>
<pre>
    2024-May-24 16:42:20.36| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 15; my-valid/acc: 0.3953488372093023; objective: 2.1200345516204835; 
    2024-May-24 16:42:20.38| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 16; objective: 2.0746865272521973; 
    2024-May-24 16:42:20.39| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 17; objective: 2.2186732292175293; 
    2024-May-24 16:42:20.40| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 18; objective: 2.0799942016601562; 
    2024-May-24 16:42:20.41| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 19; objective: 1.7716232538223267; 
    2024-May-24 16:42:20.41| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 20; objective: 1.8272186517715454; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 683.64it/s]

</pre>
<pre>
    2024-May-24 16:42:20.50| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 20; my-valid/acc: 0.4186046511627907; objective: 1.9862218618392944; 
    2024-May-24 16:42:20.52| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 21; objective: 1.957564353942871; 
    2024-May-24 16:42:20.53| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 22; objective: 1.8028621673583984; 
    2024-May-24 16:42:20.53| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 23; objective: 1.975327491760254; 
    2024-May-24 16:42:20.54| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 24; objective: 1.807953119277954; 
    2024-May-24 16:42:20.55| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 25; objective: 1.7858139276504517; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 634.74it/s]

</pre>
<pre>
    2024-May-24 16:42:20.64| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 25; my-valid/acc: 0.4883720930232558; objective: 1.8099432945251466; 
    2024-May-24 16:42:20.66| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 26; objective: 1.5425803661346436; 
    2024-May-24 16:42:20.67| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 27; objective: 1.4694782495498657; 
    2024-May-24 16:42:20.68| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 28; objective: 1.8224706649780273; 
    2024-May-24 16:42:20.68| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 29; objective: 1.5353931188583374; 
    2024-May-24 16:42:20.69| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 30; objective: 1.9406465291976929; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 748.24it/s]

</pre>
<pre>
    2024-May-24 16:42:20.78| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 30; my-valid/acc: 0.5116279069767442; objective: 1.6320355653762817; 
    2024-May-24 16:42:20.80| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 31; objective: 1.4210201501846313; 
    2024-May-24 16:42:20.81| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 32; objective: 1.8781251907348633; 
    2024-May-24 16:42:20.82| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 33; objective: 1.166929841041565; 
    2024-May-24 16:42:20.83| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 34; objective: 1.6172298192977905; 
    2024-May-24 16:42:20.84| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 35; objective: 1.5352650880813599; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 759.59it/s]

</pre>
<pre>
    2024-May-24 16:42:20.91| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 35; my-valid/acc: 0.6744186046511628; objective: 1.450360083580017; 
    2024-May-24 16:42:20.93| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 36; objective: 1.544428825378418; 
    2024-May-24 16:42:20.94| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 37; objective: 1.4825594425201416; 
    2024-May-24 16:42:20.95| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 38; objective: 1.481727957725525; 
    2024-May-24 16:42:20.96| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 39; objective: 1.5011167526245117; 
    2024-May-24 16:42:20.96| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 40; objective: 1.2468798160552979; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 825.43it/s]

</pre>
<pre>
    2024-May-24 16:42:21.03| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 40; my-valid/acc: 0.5813953488372093; objective: 1.3515176057815552; 
    2024-May-24 16:42:21.06| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 41; objective: 1.137115240097046; 
    2024-May-24 16:42:21.07| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 42; objective: 1.2611033916473389; 
    2024-May-24 16:42:21.07| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 43; objective: 1.5822103023529053; 
    2024-May-24 16:42:21.08| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 44; objective: 0.9167646169662476; 
    2024-May-24 16:42:21.09| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 45; objective: 1.0554713010787964; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 677.50it/s]

</pre>
<pre>
    2024-May-24 16:42:21.17| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 45; my-valid/acc: 0.5813953488372093; objective: 1.2751734495162963; 
    2024-May-24 16:42:21.19| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 46; objective: 1.5230745077133179; 
    2024-May-24 16:42:21.20| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 47; objective: 0.9670822024345398; 
    2024-May-24 16:42:21.21| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 48; objective: 1.2309763431549072; 
    2024-May-24 16:42:21.22| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 49; objective: 1.4959913492202759; 
    2024-May-24 16:42:21.22| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 50; objective: 1.019989013671875; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 702.56it/s]

</pre>
<pre>
    2024-May-24 16:42:21.31| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 50; my-valid/acc: 0.5813953488372093; objective: 1.1450695395469666; 
    2024-May-24 16:42:21.33| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 51; objective: 0.9896019101142883; 
    2024-May-24 16:42:21.33| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 52; objective: 1.0472078323364258; 
    2024-May-24 16:42:21.35| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 53; objective: 0.6146770715713501; 
    2024-May-24 16:42:21.35| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 54; objective: 1.223360300064087; 
    2024-May-24 16:42:21.36| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 55; objective: 1.4324121475219727; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 624.73it/s]

</pre>
<pre>
    2024-May-24 16:42:21.45| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 55; my-valid/acc: 0.5116279069767442; objective: 1.1396703839302063; 
    2024-May-24 16:42:21.47| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 56; objective: 0.8977691531181335; 
    2024-May-24 16:42:21.48| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 57; objective: 1.013144850730896; 
    2024-May-24 16:42:21.48| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 58; objective: 0.7408015131950378; 
    2024-May-24 16:42:21.49| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 59; objective: 0.662105143070221; 
    2024-May-24 16:42:21.49| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 60; objective: 0.5859256386756897; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 735.76it/s]

</pre>
<pre>
    2024-May-24 16:42:21.57| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 60; my-valid/acc: 0.7209302325581395; objective: 0.8712847888469696; 
    2024-May-24 16:42:21.59| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 61; objective: 0.8210352063179016; 
    2024-May-24 16:42:21.60| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 62; objective: 0.8280698657035828; 
    2024-May-24 16:42:21.61| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 63; objective: 0.6546609401702881; 
    2024-May-24 16:42:21.62| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 64; objective: 0.6739475727081299; 
    2024-May-24 16:42:21.63| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 65; objective: 0.5538802146911621; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 674.51it/s]

</pre>
<pre>
    2024-May-24 16:42:21.71| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 65; my-valid/acc: 0.7441860465116279; objective: 0.8647180676460267; 
    2024-May-24 16:42:21.73| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 66; objective: 0.8211454153060913; 
    2024-May-24 16:42:21.74| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 67; objective: 0.7842769622802734; 
    2024-May-24 16:42:21.75| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 68; objective: 0.5703445672988892; 
    2024-May-24 16:42:21.76| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 69; objective: 0.8491142988204956; 
    2024-May-24 16:42:21.77| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 70; objective: 0.8757203817367554; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 702.61it/s]

</pre>
<pre>
    2024-May-24 16:42:21.85| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 70; my-valid/acc: 0.6744186046511628; objective: 0.8870090007781982; 
    2024-May-24 16:42:21.86| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 71; objective: 0.5855669975280762; 
    2024-May-24 16:42:21.87| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 72; objective: 0.3257257342338562; 
    2024-May-24 16:42:21.89| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 73; objective: 0.807861328125; 
    2024-May-24 16:42:21.89| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 74; objective: 0.6515744924545288; 
    2024-May-24 16:42:21.91| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 75; objective: 0.9471737742424011; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 776.65it/s]

</pre>
<pre>
    2024-May-24 16:42:21.98| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 75; my-valid/acc: 0.6976744186046512; objective: 0.7733974754810333; 
    2024-May-24 16:42:22.01| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 76; objective: 0.31001508235931396; 
    2024-May-24 16:42:22.02| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 77; objective: 0.672425389289856; 
    2024-May-24 16:42:22.02| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 78; objective: 0.32893723249435425; 
    2024-May-24 16:42:22.03| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 79; objective: 0.4878315031528473; 
    2024-May-24 16:42:22.04| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 80; objective: 0.18520380556583405; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 704.06it/s]

</pre>
<pre>
    2024-May-24 16:42:22.12| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 80; my-valid/acc: 0.627906976744186; objective: 0.8145189821720124; 
    2024-May-24 16:42:22.13| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 81; objective: 0.8917444348335266; 
    2024-May-24 16:42:22.14| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 82; objective: 0.7088661193847656; 
    2024-May-24 16:42:22.15| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 83; objective: 0.871364951133728; 
    2024-May-24 16:42:22.16| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 84; objective: 0.565614640712738; 
    2024-May-24 16:42:22.16| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 85; objective: 1.1912280321121216; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 707.27it/s]

</pre>
<pre>
    2024-May-24 16:42:22.24| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 85; my-valid/acc: 0.7674418604651163; objective: 0.6921847283840179; 
    2024-May-24 16:42:22.26| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 86; objective: 0.8988658785820007; 
    2024-May-24 16:42:22.26| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 87; objective: 0.25610730051994324; 
    2024-May-24 16:42:22.27| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 88; objective: 0.4643763601779938; 
    2024-May-24 16:42:22.28| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 89; objective: 0.4465492367744446; 
    2024-May-24 16:42:22.29| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 90; objective: 0.3234078288078308; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 710.99it/s]

</pre>
<pre>
    2024-May-24 16:42:22.36| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 90; my-valid/acc: 0.6744186046511628; objective: 0.8348591446876525; 
    2024-May-24 16:42:22.39| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 91; objective: 0.39745527505874634; 
    2024-May-24 16:42:22.41| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 92; objective: 0.8532809019088745; 
    2024-May-24 16:42:22.42| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 93; objective: 0.47992992401123047; 
    2024-May-24 16:42:22.43| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 94; objective: 0.0838661640882492; 
    2024-May-24 16:42:22.43| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 95; objective: 0.49022263288497925; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 738.66it/s]

</pre>
<pre>
    2024-May-24 16:42:22.52| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 95; my-valid/acc: 0.7209302325581395; objective: 0.7183641791343689; 
    2024-May-24 16:42:22.53| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 96; objective: 0.5845873355865479; 
    2024-May-24 16:42:22.53| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 97; objective: 0.9997395277023315; 
    2024-May-24 16:42:22.54| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 98; objective: 0.2860856056213379; 
    2024-May-24 16:42:22.55| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 99; objective: 0.18424224853515625; 
    2024-May-24 16:42:22.56| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 100; objective: 0.14553338289260864; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 812.67it/s]

</pre>
<pre>
    2024-May-24 16:42:22.64| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 100; my-valid/acc: 0.7906976744186046; objective: 0.5649376273155212; 
    2024-May-24 16:42:22.66| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 101; objective: 0.6807511448860168; 
    2024-May-24 16:42:22.67| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 102; objective: 0.4182300567626953; 
    2024-May-24 16:42:22.67| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 103; objective: 0.5750024914741516; 
    2024-May-24 16:42:22.68| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 104; objective: 0.3903711438179016; 
    2024-May-24 16:42:22.69| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 105; objective: 0.15528497099876404; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 892.67it/s]

</pre>
<pre>
    2024-May-24 16:42:22.76| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 105; my-valid/acc: 0.8372093023255814; objective: 0.4794299900531769; 
    2024-May-24 16:42:22.78| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 106; objective: 0.4399721622467041; 
    2024-May-24 16:42:22.79| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 107; objective: 0.9834358096122742; 
    2024-May-24 16:42:22.80| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 108; objective: 0.3158140182495117; 
    2024-May-24 16:42:22.81| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 109; objective: 0.26788243651390076; 
    2024-May-24 16:42:22.81| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 110; objective: 0.38584303855895996; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 680.61it/s]

</pre>
<pre>
    2024-May-24 16:42:22.90| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 110; my-valid/acc: 0.813953488372093; objective: 0.4963296115398407; 
    2024-May-24 16:42:22.91| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 111; objective: 0.4139387011528015; 
    2024-May-24 16:42:22.91| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 112; objective: 0.3234875798225403; 
    2024-May-24 16:42:22.92| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 113; objective: 0.30029189586639404; 
    2024-May-24 16:42:22.93| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 114; objective: 0.47261935472488403; 
    2024-May-24 16:42:22.93| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 115; objective: 0.6491639614105225; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 812.68it/s]

</pre>
<pre>
    2024-May-24 16:42:23.00| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 115; my-valid/acc: 0.8372093023255814; objective: 0.46701614558696747; 
    2024-May-24 16:42:23.02| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 116; objective: 0.5192955136299133; 
    2024-May-24 16:42:23.03| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 117; objective: 0.1198667511343956; 
    2024-May-24 16:42:23.04| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 118; objective: 0.8839007616043091; 
    2024-May-24 16:42:23.05| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 119; objective: 0.6286594271659851; 
    2024-May-24 16:42:23.06| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 120; objective: 0.2967942953109741; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 807.81it/s]

</pre>
<pre>
    2024-May-24 16:42:23.13| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 120; my-valid/acc: 0.7674418604651163; objective: 0.5416155338287354; 
    2024-May-24 16:42:23.14| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 121; objective: 0.3840293288230896; 
    2024-May-24 16:42:23.15| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 122; objective: 0.26465147733688354; 
    2024-May-24 16:42:23.16| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 123; objective: 0.20785360038280487; 
    2024-May-24 16:42:23.16| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 124; objective: 0.5857481360435486; 
    2024-May-24 16:42:23.17| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 125; objective: 0.2513807415962219; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 864.08it/s]

</pre>
<pre>
    2024-May-24 16:42:23.24| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 125; my-valid/acc: 0.7441860465116279; objective: 0.6432130992412567; 
    2024-May-24 16:42:23.25| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 126; objective: 0.48046112060546875; 
    2024-May-24 16:42:23.26| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 127; objective: 0.1669985055923462; 
    2024-May-24 16:42:23.27| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 128; objective: 0.2551296353340149; 
    2024-May-24 16:42:23.27| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 129; objective: 0.39759451150894165; 
    2024-May-24 16:42:23.28| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 130; objective: 0.12379683554172516; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 901.30it/s]

</pre>
<pre>
    2024-May-24 16:42:23.35| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 130; my-valid/acc: 0.7674418604651163; objective: 0.594197279214859; 
    2024-May-24 16:42:23.36| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 131; objective: 0.2534908354282379; 
    2024-May-24 16:42:23.36| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 132; objective: 0.5093891620635986; 
    2024-May-24 16:42:23.37| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 133; objective: 0.1864108145236969; 
    2024-May-24 16:42:23.38| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 134; objective: 0.2897171080112457; 
    2024-May-24 16:42:23.38| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 135; objective: 0.5396483540534973; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 923.68it/s]

</pre>
<pre>
    2024-May-24 16:42:23.45| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 135; my-valid/acc: 0.813953488372093; objective: 0.5497540190815926; 
    2024-May-24 16:42:23.45| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 136; objective: 0.25071924924850464; 
    2024-May-24 16:42:23.46| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 137; objective: 0.5670820474624634; 
    2024-May-24 16:42:23.47| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 138; objective: 0.31639617681503296; 
    2024-May-24 16:42:23.48| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 139; objective: 0.20214490592479706; 
    2024-May-24 16:42:23.48| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: TRAIN; iteration: 140; objective: 0.32829517126083374; 

</pre>
<pre>
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:00\<00:00, 758.92it/s]
</pre>
<pre>
    2024-May-24 16:42:23.56| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:220  | fold: VALID; iteration: 140; my-valid/acc: 0.813953488372093; objective: 0.5148731887340545; 
    2024-May-24 16:42:23.56| INFO     | Duncans-MacBook-Pro.fritz.box| superduperdb.ext.torch.training:194  | early stopping triggered!
    2024-May-24 16:42:23.56| SUCCESS  | Duncans-MacBook-Pro.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x151f9e8d0\>.  function:\<function method_job at 0x110261d00\> future:23a0532c-53bb-475a-a0a4-e9e3097b485b

</pre>
<pre>
    

</pre>
</details>

The trained model is now available via `db.load` - the `model.trainer` object contains the metric traces
logged during training.

```python
from matplotlib import pyplot as plt

# Load the model from the database
model = db.load('model', model.identifier)

# Plot the accuracy values
plt.plot(model.trainer.metric_values['my-valid/acc'])
plt.show()
```

<details>
<summary>Outputs</summary>
<div>![](/training/14_0.png)</div>
</details>
