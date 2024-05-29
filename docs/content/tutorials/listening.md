
# Listening for new data

:::note
In SuperDuperDB, AI models may be configured to listen for newly inserted data.
Outputs will be computed over that data and saved back to the data-backend.
:::

In this example we show how to configure 3 models to interact when new data is added.

1. A featurizing computer vision model (images `->` vectors).
1. 2 models evaluating image-2-text similarity to a set of key-words.

We use an open-source model "CLIP" which we install via `pip` directly from GitHub.
You can read more about installing requirements on our docs [here](../get_started/environment).

```python
!pip install git+https://github.com/openai/CLIP.git
```

<details>
<summary>Outputs</summary>
<pre>
    Collecting git+https://github.com/openai/CLIP.git
      Cloning https://github.com/openai/CLIP.git to /private/var/folders/3h/p6qzszds1c7gtbmt_2qq0tvm0000gn/T/pip-req-build-spx4v54y
      Running command git clone --filter=blob:none --quiet https://github.com/openai/CLIP.git /private/var/folders/3h/p6qzszds1c7gtbmt_2qq0tvm0000gn/T/pip-req-build-spx4v54y
      Resolved https://github.com/openai/CLIP.git to commit a1d071733d7111c9c014f024669f959182114e33
      Installing build dependencies ... [?25ldone
    [?25h  Getting requirements to build wheel ... [?25ldone
    [?25h  Preparing metadata (pyproject.toml) ... [?25ldone
    [?25hRequirement already satisfied: ftfy in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from clip==1.0) (6.2.0)
    Requirement already satisfied: regex in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from clip==1.0) (2023.12.25)
    Requirement already satisfied: tqdm in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from clip==1.0) (4.66.2)
    Requirement already satisfied: torch in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from clip==1.0) (2.2.2)
    Requirement already satisfied: torchvision in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from clip==1.0) (0.17.2)
    Requirement already satisfied: wcwidth\<0.3.0,\>=0.2.12 in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from ftfy-\>clip==1.0) (0.2.13)
    Requirement already satisfied: filelock in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from torch-\>clip==1.0) (3.13.3)
    Requirement already satisfied: typing-extensions\>=4.8.0 in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from torch-\>clip==1.0) (4.11.0)
    Requirement already satisfied: sympy in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from torch-\>clip==1.0) (1.12)
    Requirement already satisfied: networkx in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from torch-\>clip==1.0) (3.3)
    Requirement already satisfied: jinja2 in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from torch-\>clip==1.0) (3.1.3)
    Requirement already satisfied: fsspec in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from torch-\>clip==1.0) (2024.2.0)
    Requirement already satisfied: numpy in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from torchvision-\>clip==1.0) (1.26.4)
    Requirement already satisfied: pillow!=8.3.*,\>=5.3.0 in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from torchvision-\>clip==1.0) (10.3.0)
    Requirement already satisfied: MarkupSafe\>=2.0 in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from jinja2-\>torch-\>clip==1.0) (2.1.5)
    Requirement already satisfied: mpmath\>=0.19 in /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages (from sympy-\>torch-\>clip==1.0) (1.3.0)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.2.1[0m[39;49m -\> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m

</pre>
</details>

We apply our setup to images from the 
[cats and dogs dataset](https://www.kaggle.com/c/dogs-vs-cats). We've prepared a subset especially 
for quick experimentation.

```python
!curl -O https://superduperdb-public-demo.s3.amazonaws.com/images.zip && unzip images.zip
from PIL import Image
import os

data = [f'images/{x}' for x in os.listdir('./images') if x.endswith('png')]
data = [{'img': Image.open(path)} for path in data]
```

<details>
<summary>Outputs</summary>

</details>

Now that we've prepared these records we can insert this data "directly" into the database with 
a standard MongoDB insert statement. (Notice however the difference from `pymongo` with the `.execute()` call.)

```python
from superduperdb import superduper, Document

db = superduper('mongomock://')

db['images'].insert_many([Document(r) for r in data[:-1]]).execute()
```

<details>
<summary>Outputs</summary>
<pre>
    2024-May-27 11:33:18.45| INFO     | Duncans-MBP.fritz.box| superduperdb.base.build:69   | Data Client is ready. mongomock.MongoClient('localhost', 27017)
    2024-May-27 11:33:18.47| INFO     | Duncans-MBP.fritz.box| superduperdb.base.build:42   | Connecting to Metadata Client with engine:  mongomock.MongoClient('localhost', 27017)
    2024-May-27 11:33:18.47| INFO     | Duncans-MBP.fritz.box| superduperdb.base.build:155  | Connecting to compute client: None
    2024-May-27 11:33:18.47| INFO     | Duncans-MBP.fritz.box| superduperdb.base.datalayer:85   | Building Data Layer
    2024-May-27 11:33:18.47| INFO     | Duncans-MBP.fritz.box| superduperdb.base.build:220  | Configuration: 
     +---------------+--------------+
    | Configuration |    Value     |
    +---------------+--------------+
    |  Data Backend | mongomock:// |
    +---------------+--------------+
    2024-May-27 11:33:18.49| WARNING  | Duncans-MBP.fritz.box| superduperdb.misc.annotations:117  | add is deprecated and will be removed in a future release.
    2024-May-27 11:33:18.49| WARNING  | Duncans-MBP.fritz.box| superduperdb.backends.local.artifacts:82   | File /tmp/e6eb888f0b0fbbab905029cb309537b9383919a6 already exists
    2024-May-27 11:33:18.49| WARNING  | Duncans-MBP.fritz.box| superduperdb.backends.local.artifacts:82   | File /tmp/ee1a946181f065af29a3c8637b2858b153d8fc8e already exists
    2024-May-27 11:34:10.59| INFO     | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function callable_job at 0x10cd4be20\>
    2024-May-27 11:34:10.99| SUCCESS  | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x13ecc76d0\>.  function:\<function callable_job at 0x10cd4be20\> future:cc9d945c-c1d2-474f-a869-a96dac41a32a

</pre>
<pre>
    ([ObjectId('665453924260c60bfe3e8084'),
      ObjectId('665453924260c60bfe3e8085'),
      ObjectId('665453924260c60bfe3e8086'),
      ObjectId('665453924260c60bfe3e8087'),
      ObjectId('665453924260c60bfe3e8088'),
      ObjectId('665453924260c60bfe3e8089'),
      ObjectId('665453924260c60bfe3e808a'),
      ObjectId('665453924260c60bfe3e808b'),
      ObjectId('665453924260c60bfe3e808c'),
      ObjectId('665453924260c60bfe3e808d'),
      ObjectId('665453924260c60bfe3e808e'),
      ObjectId('665453924260c60bfe3e808f'),
      ObjectId('665453924260c60bfe3e8090'),
      ObjectId('665453924260c60bfe3e8091'),
      ObjectId('665453924260c60bfe3e8092'),
      ObjectId('665453924260c60bfe3e8093'),
      ObjectId('665453924260c60bfe3e8094'),
      ObjectId('665453924260c60bfe3e8095'),
      ObjectId('665453924260c60bfe3e8096'),
      ObjectId('665453924260c60bfe3e8097'),
      ObjectId('665453924260c60bfe3e8098'),
      ObjectId('665453924260c60bfe3e8099'),
      ObjectId('665453924260c60bfe3e809a'),
      ObjectId('665453924260c60bfe3e809b'),
      ObjectId('665453924260c60bfe3e809c'),
      ObjectId('665453924260c60bfe3e809d'),
      ObjectId('665453924260c60bfe3e809e'),
      ObjectId('665453924260c60bfe3e809f'),
      ObjectId('665453924260c60bfe3e80a0'),
      ObjectId('665453924260c60bfe3e80a1'),
      ObjectId('665453924260c60bfe3e80a2'),
      ObjectId('665453924260c60bfe3e80a3'),
      ObjectId('665453924260c60bfe3e80a4'),
      ObjectId('665453924260c60bfe3e80a5'),
      ObjectId('665453924260c60bfe3e80a6'),
      ObjectId('665453924260c60bfe3e80a7'),
      ObjectId('665453924260c60bfe3e80a8'),
      ObjectId('665453924260c60bfe3e80a9'),
      ObjectId('665453924260c60bfe3e80aa'),
      ObjectId('665453924260c60bfe3e80ab'),
      ObjectId('665453924260c60bfe3e80ac'),
      ObjectId('665453924260c60bfe3e80ad'),
      ObjectId('665453924260c60bfe3e80ae'),
      ObjectId('665453924260c60bfe3e80af'),
      ObjectId('665453924260c60bfe3e80b0'),
      ObjectId('665453924260c60bfe3e80b1'),
      ObjectId('665453924260c60bfe3e80b2'),
      ObjectId('665453924260c60bfe3e80b3'),
      ObjectId('665453924260c60bfe3e80b4'),
      ObjectId('665453924260c60bfe3e80b5'),
      ObjectId('665453924260c60bfe3e80b6'),
      ObjectId('665453924260c60bfe3e80b7'),
      ObjectId('665453924260c60bfe3e80b8'),
      ObjectId('665453924260c60bfe3e80b9'),
      ObjectId('665453924260c60bfe3e80ba'),
      ObjectId('665453924260c60bfe3e80bb'),
      ObjectId('665453924260c60bfe3e80bc'),
      ObjectId('665453924260c60bfe3e80bd'),
      ObjectId('665453924260c60bfe3e80be'),
      ObjectId('665453924260c60bfe3e80bf'),
      ObjectId('665453924260c60bfe3e80c0'),
      ObjectId('665453924260c60bfe3e80c1'),
      ObjectId('665453924260c60bfe3e80c2'),
      ObjectId('665453924260c60bfe3e80c3'),
      ObjectId('665453924260c60bfe3e80c4'),
      ObjectId('665453924260c60bfe3e80c5'),
      ObjectId('665453924260c60bfe3e80c6'),
      ObjectId('665453924260c60bfe3e80c7'),
      ObjectId('665453924260c60bfe3e80c8'),
      ObjectId('665453924260c60bfe3e80c9'),
      ObjectId('665453924260c60bfe3e80ca'),
      ObjectId('665453924260c60bfe3e80cb'),
      ObjectId('665453924260c60bfe3e80cc'),
      ObjectId('665453924260c60bfe3e80cd'),
      ObjectId('665453924260c60bfe3e80ce'),
      ObjectId('665453924260c60bfe3e80cf'),
      ObjectId('665453924260c60bfe3e80d0'),
      ObjectId('665453924260c60bfe3e80d1'),
      ObjectId('665453924260c60bfe3e80d2'),
      ObjectId('665453924260c60bfe3e80d3'),
      ObjectId('665453924260c60bfe3e80d4'),
      ObjectId('665453924260c60bfe3e80d5'),
      ObjectId('665453924260c60bfe3e80d6'),
      ObjectId('665453924260c60bfe3e80d7'),
      ObjectId('665453924260c60bfe3e80d8'),
      ObjectId('665453924260c60bfe3e80d9'),
      ObjectId('665453924260c60bfe3e80da'),
      ObjectId('665453924260c60bfe3e80db'),
      ObjectId('665453924260c60bfe3e80dc'),
      ObjectId('665453924260c60bfe3e80dd'),
      ObjectId('665453924260c60bfe3e80de'),
      ObjectId('665453924260c60bfe3e80df'),
      ObjectId('665453924260c60bfe3e80e0'),
      ObjectId('665453924260c60bfe3e80e1'),
      ObjectId('665453924260c60bfe3e80e2'),
      ObjectId('665453924260c60bfe3e80e3'),
      ObjectId('665453924260c60bfe3e80e4'),
      ObjectId('665453924260c60bfe3e80e5'),
      ObjectId('665453924260c60bfe3e80e6'),
      ObjectId('665453924260c60bfe3e80e7'),
      ObjectId('665453924260c60bfe3e80e8'),
      ObjectId('665453924260c60bfe3e80e9'),
      ObjectId('665453924260c60bfe3e80ea'),
      ObjectId('665453924260c60bfe3e80eb'),
      ObjectId('665453924260c60bfe3e80ec'),
      ObjectId('665453924260c60bfe3e80ed'),
      ObjectId('665453924260c60bfe3e80ee'),
      ObjectId('665453924260c60bfe3e80ef'),
      ObjectId('665453924260c60bfe3e80f0'),
      ObjectId('665453924260c60bfe3e80f1'),
      ObjectId('665453924260c60bfe3e80f2'),
      ObjectId('665453924260c60bfe3e80f3'),
      ObjectId('665453924260c60bfe3e80f4'),
      ObjectId('665453924260c60bfe3e80f5'),
      ObjectId('665453924260c60bfe3e80f6'),
      ObjectId('665453924260c60bfe3e80f7'),
      ObjectId('665453924260c60bfe3e80f8'),
      ObjectId('665453924260c60bfe3e80f9'),
      ObjectId('665453924260c60bfe3e80fa'),
      ObjectId('665453924260c60bfe3e80fb'),
      ObjectId('665453924260c60bfe3e80fc'),
      ObjectId('665453924260c60bfe3e80fd'),
      ObjectId('665453924260c60bfe3e80fe'),
      ObjectId('665453924260c60bfe3e80ff'),
      ObjectId('665453924260c60bfe3e8100'),
      ObjectId('665453924260c60bfe3e8101'),
      ObjectId('665453924260c60bfe3e8102'),
      ObjectId('665453924260c60bfe3e8103'),
      ObjectId('665453924260c60bfe3e8104'),
      ObjectId('665453924260c60bfe3e8105'),
      ObjectId('665453924260c60bfe3e8106'),
      ObjectId('665453924260c60bfe3e8107'),
      ObjectId('665453924260c60bfe3e8108'),
      ObjectId('665453924260c60bfe3e8109'),
      ObjectId('665453924260c60bfe3e810a'),
      ObjectId('665453924260c60bfe3e810b'),
      ObjectId('665453924260c60bfe3e810c'),
      ObjectId('665453924260c60bfe3e810d'),
      ObjectId('665453924260c60bfe3e810e'),
      ObjectId('665453924260c60bfe3e810f'),
      ObjectId('665453924260c60bfe3e8110'),
      ObjectId('665453924260c60bfe3e8111'),
      ObjectId('665453924260c60bfe3e8112'),
      ObjectId('665453924260c60bfe3e8113'),
      ObjectId('665453924260c60bfe3e8114'),
      ObjectId('665453924260c60bfe3e8115'),
      ObjectId('665453924260c60bfe3e8116'),
      ObjectId('665453924260c60bfe3e8117'),
      ObjectId('665453924260c60bfe3e8118'),
      ObjectId('665453924260c60bfe3e8119'),
      ObjectId('665453924260c60bfe3e811a'),
      ObjectId('665453924260c60bfe3e811b'),
      ObjectId('665453924260c60bfe3e811c'),
      ObjectId('665453924260c60bfe3e811d'),
      ObjectId('665453924260c60bfe3e811e'),
      ObjectId('665453924260c60bfe3e811f'),
      ObjectId('665453924260c60bfe3e8120'),
      ObjectId('665453924260c60bfe3e8121'),
      ObjectId('665453924260c60bfe3e8122'),
      ObjectId('665453924260c60bfe3e8123'),
      ObjectId('665453924260c60bfe3e8124'),
      ObjectId('665453924260c60bfe3e8125'),
      ObjectId('665453924260c60bfe3e8126'),
      ObjectId('665453924260c60bfe3e8127'),
      ObjectId('665453924260c60bfe3e8128'),
      ObjectId('665453924260c60bfe3e8129'),
      ObjectId('665453924260c60bfe3e812a'),
      ObjectId('665453924260c60bfe3e812b'),
      ObjectId('665453924260c60bfe3e812c'),
      ObjectId('665453924260c60bfe3e812d'),
      ObjectId('665453924260c60bfe3e812e'),
      ObjectId('665453924260c60bfe3e812f'),
      ObjectId('665453924260c60bfe3e8130'),
      ObjectId('665453924260c60bfe3e8131'),
      ObjectId('665453924260c60bfe3e8132'),
      ObjectId('665453924260c60bfe3e8133'),
      ObjectId('665453924260c60bfe3e8134'),
      ObjectId('665453924260c60bfe3e8135'),
      ObjectId('665453924260c60bfe3e8136'),
      ObjectId('665453924260c60bfe3e8137'),
      ObjectId('665453924260c60bfe3e8138'),
      ObjectId('665453924260c60bfe3e8139'),
      ObjectId('665453924260c60bfe3e813a'),
      ObjectId('665453924260c60bfe3e813b'),
      ObjectId('665453924260c60bfe3e813c'),
      ObjectId('665453924260c60bfe3e813d'),
      ObjectId('665453924260c60bfe3e813e'),
      ObjectId('665453924260c60bfe3e813f'),
      ObjectId('665453924260c60bfe3e8140'),
      ObjectId('665453924260c60bfe3e8141'),
      ObjectId('665453924260c60bfe3e8142'),
      ObjectId('665453924260c60bfe3e8143'),
      ObjectId('665453924260c60bfe3e8144'),
      ObjectId('665453924260c60bfe3e8145'),
      ObjectId('665453924260c60bfe3e8146'),
      ObjectId('665453924260c60bfe3e8147'),
      ObjectId('665453924260c60bfe3e8148'),
      ObjectId('665453924260c60bfe3e8149'),
      ObjectId('665453924260c60bfe3e814a'),
      ObjectId('665453924260c60bfe3e814b'),
      ObjectId('665453924260c60bfe3e814c'),
      ObjectId('665453924260c60bfe3e814d'),
      ObjectId('665453924260c60bfe3e814e'),
      ObjectId('665453924260c60bfe3e814f'),
      ObjectId('665453924260c60bfe3e8150'),
      ObjectId('665453924260c60bfe3e8151'),
      ObjectId('665453924260c60bfe3e8152'),
      ObjectId('665453924260c60bfe3e8153'),
      ObjectId('665453924260c60bfe3e8154'),
      ObjectId('665453924260c60bfe3e8155'),
      ObjectId('665453924260c60bfe3e8156'),
      ObjectId('665453924260c60bfe3e8157'),
      ObjectId('665453924260c60bfe3e8158'),
      ObjectId('665453924260c60bfe3e8159'),
      ObjectId('665453924260c60bfe3e815a'),
      ObjectId('665453924260c60bfe3e815b'),
      ObjectId('665453924260c60bfe3e815c'),
      ObjectId('665453924260c60bfe3e815d'),
      ObjectId('665453924260c60bfe3e815e'),
      ObjectId('665453924260c60bfe3e815f'),
      ObjectId('665453924260c60bfe3e8160'),
      ObjectId('665453924260c60bfe3e8161'),
      ObjectId('665453924260c60bfe3e8162'),
      ObjectId('665453924260c60bfe3e8163'),
      ObjectId('665453924260c60bfe3e8164'),
      ObjectId('665453924260c60bfe3e8165'),
      ObjectId('665453924260c60bfe3e8166'),
      ObjectId('665453924260c60bfe3e8167'),
      ObjectId('665453924260c60bfe3e8168'),
      ObjectId('665453924260c60bfe3e8169'),
      ObjectId('665453924260c60bfe3e816a'),
      ObjectId('665453924260c60bfe3e816b'),
      ObjectId('665453924260c60bfe3e816c'),
      ObjectId('665453924260c60bfe3e816d'),
      ObjectId('665453924260c60bfe3e816e'),
      ObjectId('665453924260c60bfe3e816f'),
      ObjectId('665453924260c60bfe3e8170'),
      ObjectId('665453924260c60bfe3e8171'),
      ObjectId('665453924260c60bfe3e8172'),
      ObjectId('665453924260c60bfe3e8173'),
      ObjectId('665453924260c60bfe3e8174'),
      ObjectId('665453924260c60bfe3e8175'),
      ObjectId('665453924260c60bfe3e8176'),
      ObjectId('665453924260c60bfe3e8177'),
      ObjectId('665453924260c60bfe3e8178'),
      ObjectId('665453924260c60bfe3e8179'),
      ObjectId('665453924260c60bfe3e817a'),
      ObjectId('665453924260c60bfe3e817b'),
      ObjectId('665453924260c60bfe3e817c'),
      ObjectId('665453924260c60bfe3e817d'),
      ObjectId('665453924260c60bfe3e817e'),
      ObjectId('665453924260c60bfe3e817f'),
      ObjectId('665453924260c60bfe3e8180'),
      ObjectId('665453924260c60bfe3e8181'),
      ObjectId('665453924260c60bfe3e8182'),
      ObjectId('665453924260c60bfe3e8183'),
      ObjectId('665453924260c60bfe3e8184'),
      ObjectId('665453924260c60bfe3e8185'),
      ObjectId('665453924260c60bfe3e8186'),
      ObjectId('665453924260c60bfe3e8187'),
      ObjectId('665453924260c60bfe3e8188'),
      ObjectId('665453924260c60bfe3e8189'),
      ObjectId('665453924260c60bfe3e818a'),
      ObjectId('665453924260c60bfe3e818b'),
      ObjectId('665453924260c60bfe3e818c'),
      ObjectId('665453924260c60bfe3e818d'),
      ObjectId('665453924260c60bfe3e818e'),
      ObjectId('665453924260c60bfe3e818f'),
      ObjectId('665453924260c60bfe3e8190'),
      ObjectId('665453924260c60bfe3e8191'),
      ObjectId('665453924260c60bfe3e8192'),
      ObjectId('665453924260c60bfe3e8193'),
      ObjectId('665453924260c60bfe3e8194'),
      ObjectId('665453924260c60bfe3e8195'),
      ObjectId('665453924260c60bfe3e8196'),
      ObjectId('665453924260c60bfe3e8197'),
      ObjectId('665453924260c60bfe3e8198'),
      ObjectId('665453924260c60bfe3e8199'),
      ObjectId('665453924260c60bfe3e819a'),
      ObjectId('665453924260c60bfe3e819b'),
      ObjectId('665453924260c60bfe3e819c'),
      ObjectId('665453924260c60bfe3e819d'),
      ObjectId('665453924260c60bfe3e819e'),
      ObjectId('665453924260c60bfe3e819f'),
      ObjectId('665453924260c60bfe3e81a0'),
      ObjectId('665453924260c60bfe3e81a1'),
      ObjectId('665453924260c60bfe3e81a2'),
      ObjectId('665453924260c60bfe3e81a3'),
      ObjectId('665453924260c60bfe3e81a4'),
      ObjectId('665453924260c60bfe3e81a5'),
      ObjectId('665453924260c60bfe3e81a6'),
      ObjectId('665453924260c60bfe3e81a7'),
      ObjectId('665453924260c60bfe3e81a8'),
      ObjectId('665453924260c60bfe3e81a9'),
      ObjectId('665453924260c60bfe3e81aa'),
      ObjectId('665453924260c60bfe3e81ab'),
      ObjectId('665453924260c60bfe3e81ac'),
      ObjectId('665453924260c60bfe3e81ad'),
      ObjectId('665453924260c60bfe3e81ae'),
      ObjectId('665453924260c60bfe3e81af'),
      ObjectId('665453924260c60bfe3e81b0'),
      ObjectId('665453924260c60bfe3e81b1'),
      ObjectId('665453924260c60bfe3e81b2'),
      ObjectId('665453924260c60bfe3e81b3'),
      ObjectId('665453924260c60bfe3e81b4'),
      ObjectId('665453924260c60bfe3e81b5'),
      ObjectId('665453924260c60bfe3e81b6'),
      ObjectId('665453924260c60bfe3e81b7'),
      ObjectId('665453924260c60bfe3e81b8'),
      ObjectId('665453924260c60bfe3e81b9'),
      ObjectId('665453924260c60bfe3e81ba'),
      ObjectId('665453924260c60bfe3e81bb'),
      ObjectId('665453924260c60bfe3e81bc'),
      ObjectId('665453924260c60bfe3e81bd'),
      ObjectId('665453924260c60bfe3e81be'),
      ObjectId('665453924260c60bfe3e81bf'),
      ObjectId('665453924260c60bfe3e81c0'),
      ObjectId('665453924260c60bfe3e81c1'),
      ObjectId('665453924260c60bfe3e81c2'),
      ObjectId('665453924260c60bfe3e81c3'),
      ObjectId('665453924260c60bfe3e81c4'),
      ObjectId('665453924260c60bfe3e81c5'),
      ObjectId('665453924260c60bfe3e81c6'),
      ObjectId('665453924260c60bfe3e81c7'),
      ObjectId('665453924260c60bfe3e81c8'),
      ObjectId('665453924260c60bfe3e81c9'),
      ObjectId('665453924260c60bfe3e81ca'),
      ObjectId('665453924260c60bfe3e81cb'),
      ObjectId('665453924260c60bfe3e81cc'),
      ObjectId('665453924260c60bfe3e81cd'),
      ObjectId('665453924260c60bfe3e81ce'),
      ObjectId('665453924260c60bfe3e81cf'),
      ObjectId('665453924260c60bfe3e81d0'),
      ObjectId('665453924260c60bfe3e81d1'),
      ObjectId('665453924260c60bfe3e81d2'),
      ObjectId('665453924260c60bfe3e81d3'),
      ObjectId('665453924260c60bfe3e81d4'),
      ObjectId('665453924260c60bfe3e81d5'),
      ObjectId('665453924260c60bfe3e81d6'),
      ObjectId('665453924260c60bfe3e81d7'),
      ObjectId('665453924260c60bfe3e81d8'),
      ObjectId('665453924260c60bfe3e81d9'),
      ObjectId('665453924260c60bfe3e81da'),
      ObjectId('665453924260c60bfe3e81db'),
      ObjectId('665453924260c60bfe3e81dc'),
      ObjectId('665453924260c60bfe3e81dd'),
      ObjectId('665453924260c60bfe3e81de'),
      ObjectId('665453924260c60bfe3e81df'),
      ObjectId('665453924260c60bfe3e81e0'),
      ObjectId('665453924260c60bfe3e81e1'),
      ObjectId('665453924260c60bfe3e81e2'),
      ObjectId('665453924260c60bfe3e81e3'),
      ObjectId('665453924260c60bfe3e81e4'),
      ObjectId('665453924260c60bfe3e81e5'),
      ObjectId('665453924260c60bfe3e81e6'),
      ObjectId('665453924260c60bfe3e81e7'),
      ObjectId('665453924260c60bfe3e81e8'),
      ObjectId('665453924260c60bfe3e81e9'),
      ObjectId('665453924260c60bfe3e81ea'),
      ObjectId('665453924260c60bfe3e81eb'),
      ObjectId('665453924260c60bfe3e81ec'),
      ObjectId('665453924260c60bfe3e81ed'),
      ObjectId('665453924260c60bfe3e81ee'),
      ObjectId('665453924260c60bfe3e81ef'),
      ObjectId('665453924260c60bfe3e81f0'),
      ObjectId('665453924260c60bfe3e81f1'),
      ObjectId('665453924260c60bfe3e81f2'),
      ObjectId('665453924260c60bfe3e81f3'),
      ObjectId('665453924260c60bfe3e81f4'),
      ObjectId('665453924260c60bfe3e81f5'),
      ObjectId('665453924260c60bfe3e81f6'),
      ObjectId('665453924260c60bfe3e81f7'),
      ObjectId('665453924260c60bfe3e81f8'),
      ObjectId('665453924260c60bfe3e81f9'),
      ObjectId('665453924260c60bfe3e81fa'),
      ObjectId('665453924260c60bfe3e81fb'),
      ObjectId('665453924260c60bfe3e81fc'),
      ObjectId('665453924260c60bfe3e81fd'),
      ObjectId('665453924260c60bfe3e81fe'),
      ObjectId('665453924260c60bfe3e81ff'),
      ObjectId('665453924260c60bfe3e8200'),
      ObjectId('665453924260c60bfe3e8201'),
      ObjectId('665453924260c60bfe3e8202'),
      ObjectId('665453924260c60bfe3e8203'),
      ObjectId('665453924260c60bfe3e8204'),
      ObjectId('665453924260c60bfe3e8205'),
      ObjectId('665453924260c60bfe3e8206'),
      ObjectId('665453924260c60bfe3e8207'),
      ObjectId('665453924260c60bfe3e8208'),
      ObjectId('665453924260c60bfe3e8209'),
      ObjectId('665453924260c60bfe3e820a'),
      ObjectId('665453924260c60bfe3e820b'),
      ObjectId('665453924260c60bfe3e820c'),
      ObjectId('665453924260c60bfe3e820d'),
      ObjectId('665453924260c60bfe3e820e'),
      ObjectId('665453924260c60bfe3e820f'),
      ObjectId('665453924260c60bfe3e8210'),
      ObjectId('665453924260c60bfe3e8211'),
      ObjectId('665453924260c60bfe3e8212'),
      ObjectId('665453924260c60bfe3e8213'),
      ObjectId('665453924260c60bfe3e8214'),
      ObjectId('665453924260c60bfe3e8215'),
      ObjectId('665453924260c60bfe3e8216'),
      ObjectId('665453924260c60bfe3e8217'),
      ObjectId('665453924260c60bfe3e8218'),
      ObjectId('665453924260c60bfe3e8219'),
      ObjectId('665453924260c60bfe3e821a'),
      ObjectId('665453924260c60bfe3e821b'),
      ObjectId('665453924260c60bfe3e821c'),
      ObjectId('665453924260c60bfe3e821d'),
      ObjectId('665453924260c60bfe3e821e'),
      ObjectId('665453924260c60bfe3e821f'),
      ObjectId('665453924260c60bfe3e8220'),
      ObjectId('665453924260c60bfe3e8221'),
      ObjectId('665453924260c60bfe3e8222'),
      ObjectId('665453924260c60bfe3e8223'),
      ObjectId('665453924260c60bfe3e8224'),
      ObjectId('665453924260c60bfe3e8225'),
      ObjectId('665453924260c60bfe3e8226'),
      ObjectId('665453924260c60bfe3e8227'),
      ObjectId('665453924260c60bfe3e8228'),
      ObjectId('665453924260c60bfe3e8229'),
      ObjectId('665453924260c60bfe3e822a'),
      ObjectId('665453924260c60bfe3e822b'),
      ObjectId('665453924260c60bfe3e822c'),
      ObjectId('665453924260c60bfe3e822d'),
      ObjectId('665453924260c60bfe3e822e'),
      ObjectId('665453924260c60bfe3e822f'),
      ObjectId('665453924260c60bfe3e8230'),
      ObjectId('665453924260c60bfe3e8231'),
      ObjectId('665453924260c60bfe3e8232'),
      ObjectId('665453924260c60bfe3e8233'),
      ObjectId('665453924260c60bfe3e8234'),
      ObjectId('665453924260c60bfe3e8235'),
      ObjectId('665453924260c60bfe3e8236'),
      ObjectId('665453924260c60bfe3e8237'),
      ObjectId('665453924260c60bfe3e8238'),
      ObjectId('665453924260c60bfe3e8239'),
      ObjectId('665453924260c60bfe3e823a'),
      ObjectId('665453924260c60bfe3e823b'),
      ObjectId('665453924260c60bfe3e823c'),
      ObjectId('665453924260c60bfe3e823d'),
      ObjectId('665453924260c60bfe3e823e'),
      ObjectId('665453924260c60bfe3e823f'),
      ObjectId('665453924260c60bfe3e8240'),
      ObjectId('665453924260c60bfe3e8241'),
      ObjectId('665453924260c60bfe3e8242'),
      ObjectId('665453924260c60bfe3e8243'),
      ObjectId('665453924260c60bfe3e8244'),
      ObjectId('665453924260c60bfe3e8245'),
      ObjectId('665453924260c60bfe3e8246'),
      ObjectId('665453924260c60bfe3e8247'),
      ObjectId('665453924260c60bfe3e8248'),
      ObjectId('665453924260c60bfe3e8249'),
      ObjectId('665453924260c60bfe3e824a'),
      ObjectId('665453924260c60bfe3e824b'),
      ObjectId('665453924260c60bfe3e824c'),
      ObjectId('665453924260c60bfe3e824d'),
      ObjectId('665453924260c60bfe3e824e'),
      ObjectId('665453924260c60bfe3e824f'),
      ObjectId('665453924260c60bfe3e8250'),
      ObjectId('665453924260c60bfe3e8251'),
      ObjectId('665453924260c60bfe3e8252'),
      ObjectId('665453924260c60bfe3e8253'),
      ObjectId('665453924260c60bfe3e8254'),
      ObjectId('665453924260c60bfe3e8255'),
      ObjectId('665453924260c60bfe3e8256'),
      ObjectId('665453924260c60bfe3e8257'),
      ObjectId('665453924260c60bfe3e8258'),
      ObjectId('665453924260c60bfe3e8259'),
      ObjectId('665453924260c60bfe3e825a'),
      ObjectId('665453924260c60bfe3e825b'),
      ObjectId('665453924260c60bfe3e825c'),
      ObjectId('665453924260c60bfe3e825d'),
      ObjectId('665453924260c60bfe3e825e'),
      ObjectId('665453924260c60bfe3e825f'),
      ObjectId('665453924260c60bfe3e8260'),
      ObjectId('665453924260c60bfe3e8261'),
      ObjectId('665453924260c60bfe3e8262'),
      ObjectId('665453924260c60bfe3e8263'),
      ObjectId('665453924260c60bfe3e8264'),
      ObjectId('665453924260c60bfe3e8265'),
      ObjectId('665453924260c60bfe3e8266'),
      ObjectId('665453924260c60bfe3e8267'),
      ObjectId('665453924260c60bfe3e8268'),
      ObjectId('665453924260c60bfe3e8269'),
      ObjectId('665453924260c60bfe3e826a'),
      ObjectId('665453924260c60bfe3e826b'),
      ObjectId('665453924260c60bfe3e826c'),
      ObjectId('665453924260c60bfe3e826d'),
      ObjectId('665453924260c60bfe3e826e'),
      ObjectId('665453924260c60bfe3e826f'),
      ObjectId('665453924260c60bfe3e8270'),
      ObjectId('665453924260c60bfe3e8271'),
      ObjectId('665453924260c60bfe3e8272'),
      ObjectId('665453924260c60bfe3e8273'),
      ObjectId('665453924260c60bfe3e8274'),
      ObjectId('665453924260c60bfe3e8275'),
      ObjectId('665453924260c60bfe3e8276'),
      ObjectId('665453924260c60bfe3e8277'),
      ObjectId('665453924260c60bfe3e8278'),
      ObjectId('665453924260c60bfe3e8279'),
      ObjectId('665453924260c60bfe3e827a'),
      ObjectId('665453924260c60bfe3e827b'),
      ObjectId('665453924260c60bfe3e827c'),
      ObjectId('665453924260c60bfe3e827d'),
      ObjectId('665453924260c60bfe3e827e'),
      ObjectId('665453924260c60bfe3e827f'),
      ObjectId('665453924260c60bfe3e8280'),
      ObjectId('665453924260c60bfe3e8281'),
      ObjectId('665453924260c60bfe3e8282'),
      ObjectId('665453924260c60bfe3e8283'),
      ObjectId('665453924260c60bfe3e8284'),
      ObjectId('665453924260c60bfe3e8285'),
      ObjectId('665453924260c60bfe3e8286'),
      ObjectId('665453924260c60bfe3e8287'),
      ObjectId('665453924260c60bfe3e8288'),
      ObjectId('665453924260c60bfe3e8289'),
      ObjectId('665453924260c60bfe3e828a'),
      ObjectId('665453924260c60bfe3e828b'),
      ObjectId('665453924260c60bfe3e828c'),
      ObjectId('665453924260c60bfe3e828d'),
      ObjectId('665453924260c60bfe3e828e'),
      ObjectId('665453924260c60bfe3e828f'),
      ObjectId('665453924260c60bfe3e8290'),
      ObjectId('665453924260c60bfe3e8291'),
      ObjectId('665453924260c60bfe3e8292'),
      ObjectId('665453924260c60bfe3e8293'),
      ObjectId('665453924260c60bfe3e8294'),
      ObjectId('665453924260c60bfe3e8295'),
      ObjectId('665453924260c60bfe3e8296'),
      ObjectId('665453924260c60bfe3e8297'),
      ObjectId('665453924260c60bfe3e8298'),
      ObjectId('665453924260c60bfe3e8299'),
      ObjectId('665453924260c60bfe3e829a'),
      ObjectId('665453924260c60bfe3e829b'),
      ObjectId('665453924260c60bfe3e829c'),
      ObjectId('665453924260c60bfe3e829d'),
      ObjectId('665453924260c60bfe3e829e'),
      ObjectId('665453924260c60bfe3e829f'),
      ObjectId('665453924260c60bfe3e82a0'),
      ObjectId('665453924260c60bfe3e82a1'),
      ObjectId('665453924260c60bfe3e82a2'),
      ObjectId('665453924260c60bfe3e82a3'),
      ObjectId('665453924260c60bfe3e82a4'),
      ObjectId('665453924260c60bfe3e82a5'),
      ObjectId('665453924260c60bfe3e82a6'),
      ObjectId('665453924260c60bfe3e82a7'),
      ObjectId('665453924260c60bfe3e82a8'),
      ObjectId('665453924260c60bfe3e82a9'),
      ObjectId('665453924260c60bfe3e82aa'),
      ObjectId('665453924260c60bfe3e82ab'),
      ObjectId('665453924260c60bfe3e82ac'),
      ObjectId('665453924260c60bfe3e82ad'),
      ObjectId('665453924260c60bfe3e82ae'),
      ObjectId('665453924260c60bfe3e82af'),
      ObjectId('665453924260c60bfe3e82b0'),
      ObjectId('665453924260c60bfe3e82b1'),
      ObjectId('665453924260c60bfe3e82b2'),
      ObjectId('665453924260c60bfe3e82b3'),
      ObjectId('665453924260c60bfe3e82b4'),
      ObjectId('665453924260c60bfe3e82b5'),
      ObjectId('665453924260c60bfe3e82b6'),
      ObjectId('665453924260c60bfe3e82b7'),
      ObjectId('665453924260c60bfe3e82b8'),
      ObjectId('665453924260c60bfe3e82b9'),
      ObjectId('665453924260c60bfe3e82ba'),
      ObjectId('665453924260c60bfe3e82bb'),
      ObjectId('665453924260c60bfe3e82bc'),
      ObjectId('665453924260c60bfe3e82bd'),
      ObjectId('665453924260c60bfe3e82be'),
      ObjectId('665453924260c60bfe3e82bf'),
      ObjectId('665453924260c60bfe3e82c0'),
      ObjectId('665453924260c60bfe3e82c1'),
      ObjectId('665453924260c60bfe3e82c2'),
      ObjectId('665453924260c60bfe3e82c3'),
      ObjectId('665453924260c60bfe3e82c4'),
      ObjectId('665453924260c60bfe3e82c5'),
      ObjectId('665453924260c60bfe3e82c6'),
      ObjectId('665453924260c60bfe3e82c7'),
      ObjectId('665453924260c60bfe3e82c8'),
      ObjectId('665453924260c60bfe3e82c9'),
      ObjectId('665453924260c60bfe3e82ca'),
      ObjectId('665453924260c60bfe3e82cb'),
      ObjectId('665453924260c60bfe3e82cc'),
      ObjectId('665453924260c60bfe3e82cd'),
      ObjectId('665453924260c60bfe3e82ce'),
      ObjectId('665453924260c60bfe3e82cf'),
      ObjectId('665453924260c60bfe3e82d0'),
      ObjectId('665453924260c60bfe3e82d1'),
      ObjectId('665453924260c60bfe3e82d2'),
      ObjectId('665453924260c60bfe3e82d3'),
      ObjectId('665453924260c60bfe3e82d4'),
      ObjectId('665453924260c60bfe3e82d5'),
      ObjectId('665453924260c60bfe3e82d6'),
      ObjectId('665453924260c60bfe3e82d7'),
      ObjectId('665453924260c60bfe3e82d8'),
      ObjectId('665453924260c60bfe3e82d9'),
      ObjectId('665453924260c60bfe3e82da'),
      ObjectId('665453924260c60bfe3e82db'),
      ObjectId('665453924260c60bfe3e82dc'),
      ObjectId('665453924260c60bfe3e82dd'),
      ObjectId('665453924260c60bfe3e82de'),
      ObjectId('665453924260c60bfe3e82df'),
      ObjectId('665453924260c60bfe3e82e0'),
      ObjectId('665453924260c60bfe3e82e1'),
      ObjectId('665453924260c60bfe3e82e2'),
      ObjectId('665453924260c60bfe3e82e3'),
      ObjectId('665453924260c60bfe3e82e4'),
      ObjectId('665453924260c60bfe3e82e5'),
      ObjectId('665453924260c60bfe3e82e6'),
      ObjectId('665453924260c60bfe3e82e7'),
      ObjectId('665453924260c60bfe3e82e8'),
      ObjectId('665453924260c60bfe3e82e9'),
      ObjectId('665453924260c60bfe3e82ea'),
      ObjectId('665453924260c60bfe3e82eb'),
      ObjectId('665453924260c60bfe3e82ec'),
      ObjectId('665453924260c60bfe3e82ed'),
      ObjectId('665453924260c60bfe3e82ee'),
      ObjectId('665453924260c60bfe3e82ef'),
      ObjectId('665453924260c60bfe3e82f0'),
      ObjectId('665453924260c60bfe3e82f1'),
      ObjectId('665453924260c60bfe3e82f2'),
      ObjectId('665453924260c60bfe3e82f3'),
      ObjectId('665453924260c60bfe3e82f4'),
      ObjectId('665453924260c60bfe3e82f5'),
      ObjectId('665453924260c60bfe3e82f6'),
      ObjectId('665453924260c60bfe3e82f7'),
      ObjectId('665453924260c60bfe3e82f8'),
      ObjectId('665453924260c60bfe3e82f9'),
      ObjectId('665453924260c60bfe3e82fa'),
      ObjectId('665453924260c60bfe3e82fb'),
      ObjectId('665453924260c60bfe3e82fc'),
      ObjectId('665453924260c60bfe3e82fd'),
      ObjectId('665453924260c60bfe3e82fe'),
      ObjectId('665453924260c60bfe3e82ff'),
      ObjectId('665453924260c60bfe3e8300'),
      ObjectId('665453924260c60bfe3e8301'),
      ObjectId('665453924260c60bfe3e8302'),
      ObjectId('665453924260c60bfe3e8303'),
      ObjectId('665453924260c60bfe3e8304'),
      ObjectId('665453924260c60bfe3e8305'),
      ObjectId('665453924260c60bfe3e8306'),
      ObjectId('665453924260c60bfe3e8307'),
      ObjectId('665453924260c60bfe3e8308'),
      ObjectId('665453924260c60bfe3e8309'),
      ObjectId('665453924260c60bfe3e830a'),
      ObjectId('665453924260c60bfe3e830b'),
      ObjectId('665453924260c60bfe3e830c'),
      ObjectId('665453924260c60bfe3e830d'),
      ObjectId('665453924260c60bfe3e830e'),
      ObjectId('665453924260c60bfe3e830f'),
      ObjectId('665453924260c60bfe3e8310'),
      ObjectId('665453924260c60bfe3e8311'),
      ObjectId('665453924260c60bfe3e8312'),
      ObjectId('665453924260c60bfe3e8313'),
      ObjectId('665453924260c60bfe3e8314'),
      ObjectId('665453924260c60bfe3e8315'),
      ObjectId('665453924260c60bfe3e8316'),
      ObjectId('665453924260c60bfe3e8317'),
      ObjectId('665453924260c60bfe3e8318'),
      ObjectId('665453924260c60bfe3e8319'),
      ObjectId('665453924260c60bfe3e831a'),
      ObjectId('665453924260c60bfe3e831b'),
      ObjectId('665453924260c60bfe3e831c'),
      ObjectId('665453924260c60bfe3e831d'),
      ObjectId('665453924260c60bfe3e831e'),
      ObjectId('665453924260c60bfe3e831f'),
      ObjectId('665453924260c60bfe3e8320'),
      ObjectId('665453924260c60bfe3e8321'),
      ObjectId('665453924260c60bfe3e8322'),
      ObjectId('665453924260c60bfe3e8323'),
      ObjectId('665453924260c60bfe3e8324'),
      ObjectId('665453924260c60bfe3e8325'),
      ObjectId('665453924260c60bfe3e8326'),
      ObjectId('665453924260c60bfe3e8327'),
      ObjectId('665453924260c60bfe3e8328'),
      ObjectId('665453924260c60bfe3e8329'),
      ObjectId('665453924260c60bfe3e832a'),
      ObjectId('665453924260c60bfe3e832b'),
      ObjectId('665453924260c60bfe3e832c'),
      ObjectId('665453924260c60bfe3e832d'),
      ObjectId('665453924260c60bfe3e832e'),
      ObjectId('665453924260c60bfe3e832f'),
      ObjectId('665453924260c60bfe3e8330'),
      ObjectId('665453924260c60bfe3e8331'),
      ObjectId('665453924260c60bfe3e8332'),
      ObjectId('665453924260c60bfe3e8333'),
      ObjectId('665453924260c60bfe3e8334'),
      ObjectId('665453924260c60bfe3e8335'),
      ObjectId('665453924260c60bfe3e8336'),
      ObjectId('665453924260c60bfe3e8337'),
      ObjectId('665453924260c60bfe3e8338'),
      ObjectId('665453924260c60bfe3e8339'),
      ObjectId('665453924260c60bfe3e833a'),
      ObjectId('665453924260c60bfe3e833b'),
      ObjectId('665453924260c60bfe3e833c'),
      ObjectId('665453924260c60bfe3e833d'),
      ObjectId('665453924260c60bfe3e833e'),
      ObjectId('665453924260c60bfe3e833f'),
      ObjectId('665453924260c60bfe3e8340'),
      ObjectId('665453924260c60bfe3e8341'),
      ObjectId('665453924260c60bfe3e8342'),
      ObjectId('665453924260c60bfe3e8343'),
      ObjectId('665453924260c60bfe3e8344'),
      ObjectId('665453924260c60bfe3e8345'),
      ObjectId('665453924260c60bfe3e8346'),
      ObjectId('665453924260c60bfe3e8347'),
      ObjectId('665453924260c60bfe3e8348'),
      ObjectId('665453924260c60bfe3e8349'),
      ObjectId('665453924260c60bfe3e834a'),
      ObjectId('665453924260c60bfe3e834b'),
      ObjectId('665453924260c60bfe3e834c'),
      ObjectId('665453924260c60bfe3e834d'),
      ObjectId('665453924260c60bfe3e834e'),
      ObjectId('665453924260c60bfe3e834f'),
      ObjectId('665453924260c60bfe3e8350'),
      ObjectId('665453924260c60bfe3e8351'),
      ObjectId('665453924260c60bfe3e8352'),
      ObjectId('665453924260c60bfe3e8353'),
      ObjectId('665453924260c60bfe3e8354'),
      ObjectId('665453924260c60bfe3e8355'),
      ObjectId('665453924260c60bfe3e8356'),
      ObjectId('665453924260c60bfe3e8357'),
      ObjectId('665453924260c60bfe3e8358'),
      ObjectId('665453924260c60bfe3e8359'),
      ObjectId('665453924260c60bfe3e835a'),
      ObjectId('665453924260c60bfe3e835b'),
      ObjectId('665453924260c60bfe3e835c'),
      ObjectId('665453924260c60bfe3e835d'),
      ObjectId('665453924260c60bfe3e835e'),
      ObjectId('665453924260c60bfe3e835f'),
      ObjectId('665453924260c60bfe3e8360'),
      ObjectId('665453924260c60bfe3e8361'),
      ObjectId('665453924260c60bfe3e8362'),
      ObjectId('665453924260c60bfe3e8363'),
      ObjectId('665453924260c60bfe3e8364'),
      ObjectId('665453924260c60bfe3e8365'),
      ObjectId('665453924260c60bfe3e8366'),
      ObjectId('665453924260c60bfe3e8367'),
      ObjectId('665453924260c60bfe3e8368'),
      ObjectId('665453924260c60bfe3e8369'),
      ObjectId('665453924260c60bfe3e836a'),
      ObjectId('665453924260c60bfe3e836b'),
      ObjectId('665453924260c60bfe3e836c'),
      ObjectId('665453924260c60bfe3e836d'),
      ObjectId('665453924260c60bfe3e836e'),
      ObjectId('665453924260c60bfe3e836f'),
      ObjectId('665453924260c60bfe3e8370'),
      ObjectId('665453924260c60bfe3e8371'),
      ObjectId('665453924260c60bfe3e8372'),
      ObjectId('665453924260c60bfe3e8373'),
      ObjectId('665453924260c60bfe3e8374'),
      ObjectId('665453924260c60bfe3e8375'),
      ObjectId('665453924260c60bfe3e8376'),
      ObjectId('665453924260c60bfe3e8377'),
      ObjectId('665453924260c60bfe3e8378'),
      ObjectId('665453924260c60bfe3e8379'),
      ObjectId('665453924260c60bfe3e837a'),
      ObjectId('665453924260c60bfe3e837b'),
      ObjectId('665453924260c60bfe3e837c'),
      ObjectId('665453924260c60bfe3e837d'),
      ObjectId('665453924260c60bfe3e837e'),
      ObjectId('665453924260c60bfe3e837f'),
      ObjectId('665453924260c60bfe3e8380'),
      ObjectId('665453924260c60bfe3e8381'),
      ObjectId('665453924260c60bfe3e8382'),
      ObjectId('665453924260c60bfe3e8383'),
      ObjectId('665453924260c60bfe3e8384'),
      ObjectId('665453924260c60bfe3e8385'),
      ObjectId('665453924260c60bfe3e8386'),
      ObjectId('665453924260c60bfe3e8387'),
      ObjectId('665453924260c60bfe3e8388'),
      ObjectId('665453924260c60bfe3e8389'),
      ObjectId('665453924260c60bfe3e838a'),
      ObjectId('665453924260c60bfe3e838b'),
      ObjectId('665453924260c60bfe3e838c'),
      ObjectId('665453924260c60bfe3e838d'),
      ObjectId('665453924260c60bfe3e838e'),
      ObjectId('665453924260c60bfe3e838f'),
      ObjectId('665453924260c60bfe3e8390'),
      ObjectId('665453924260c60bfe3e8391'),
      ObjectId('665453924260c60bfe3e8392'),
      ObjectId('665453924260c60bfe3e8393'),
      ObjectId('665453924260c60bfe3e8394'),
      ObjectId('665453924260c60bfe3e8395'),
      ObjectId('665453924260c60bfe3e8396'),
      ObjectId('665453924260c60bfe3e8397'),
      ObjectId('665453924260c60bfe3e8398'),
      ObjectId('665453924260c60bfe3e8399'),
      ObjectId('665453924260c60bfe3e839a'),
      ObjectId('665453924260c60bfe3e839b'),
      ObjectId('665453924260c60bfe3e839c'),
      ObjectId('665453924260c60bfe3e839d'),
      ObjectId('665453924260c60bfe3e839e'),
      ObjectId('665453924260c60bfe3e839f'),
      ObjectId('665453924260c60bfe3e83a0'),
      ObjectId('665453924260c60bfe3e83a1'),
      ObjectId('665453924260c60bfe3e83a2'),
      ObjectId('665453924260c60bfe3e83a3'),
      ObjectId('665453924260c60bfe3e83a4'),
      ObjectId('665453924260c60bfe3e83a5'),
      ObjectId('665453924260c60bfe3e83a6'),
      ObjectId('665453924260c60bfe3e83a7'),
      ObjectId('665453924260c60bfe3e83a8'),
      ObjectId('665453924260c60bfe3e83a9'),
      ObjectId('665453924260c60bfe3e83aa'),
      ObjectId('665453924260c60bfe3e83ab'),
      ObjectId('665453924260c60bfe3e83ac'),
      ObjectId('665453924260c60bfe3e83ad'),
      ObjectId('665453924260c60bfe3e83ae'),
      ObjectId('665453924260c60bfe3e83af'),
      ObjectId('665453924260c60bfe3e83b0'),
      ObjectId('665453924260c60bfe3e83b1'),
      ObjectId('665453924260c60bfe3e83b2'),
      ObjectId('665453924260c60bfe3e83b3'),
      ObjectId('665453924260c60bfe3e83b4'),
      ObjectId('665453924260c60bfe3e83b5'),
      ObjectId('665453924260c60bfe3e83b6'),
      ObjectId('665453924260c60bfe3e83b7'),
      ObjectId('665453924260c60bfe3e83b8'),
      ObjectId('665453924260c60bfe3e83b9'),
      ObjectId('665453924260c60bfe3e83ba'),
      ObjectId('665453924260c60bfe3e83bb'),
      ObjectId('665453924260c60bfe3e83bc'),
      ObjectId('665453924260c60bfe3e83bd'),
      ObjectId('665453924260c60bfe3e83be'),
      ObjectId('665453924260c60bfe3e83bf'),
      ObjectId('665453924260c60bfe3e83c0'),
      ObjectId('665453924260c60bfe3e83c1'),
      ObjectId('665453924260c60bfe3e83c2'),
      ObjectId('665453924260c60bfe3e83c3'),
      ObjectId('665453924260c60bfe3e83c4'),
      ObjectId('665453924260c60bfe3e83c5'),
      ObjectId('665453924260c60bfe3e83c6'),
      ObjectId('665453924260c60bfe3e83c7'),
      ObjectId('665453924260c60bfe3e83c8'),
      ObjectId('665453924260c60bfe3e83c9'),
      ObjectId('665453924260c60bfe3e83ca'),
      ObjectId('665453924260c60bfe3e83cb'),
      ObjectId('665453924260c60bfe3e83cc'),
      ObjectId('665453924260c60bfe3e83cd'),
      ObjectId('665453924260c60bfe3e83ce'),
      ObjectId('665453924260c60bfe3e83cf'),
      ObjectId('665453924260c60bfe3e83d0'),
      ObjectId('665453924260c60bfe3e83d1'),
      ObjectId('665453924260c60bfe3e83d2'),
      ObjectId('665453924260c60bfe3e83d3'),
      ObjectId('665453924260c60bfe3e83d4'),
      ObjectId('665453924260c60bfe3e83d5'),
      ObjectId('665453924260c60bfe3e83d6'),
      ObjectId('665453924260c60bfe3e83d7'),
      ObjectId('665453924260c60bfe3e83d8'),
      ObjectId('665453924260c60bfe3e83d9'),
      ObjectId('665453924260c60bfe3e83da'),
      ObjectId('665453924260c60bfe3e83db'),
      ObjectId('665453924260c60bfe3e83dc'),
      ObjectId('665453924260c60bfe3e83dd'),
      ObjectId('665453924260c60bfe3e83de'),
      ObjectId('665453924260c60bfe3e83df'),
      ObjectId('665453924260c60bfe3e83e0'),
      ObjectId('665453924260c60bfe3e83e1'),
      ObjectId('665453924260c60bfe3e83e2'),
      ObjectId('665453924260c60bfe3e83e3'),
      ObjectId('665453924260c60bfe3e83e4'),
      ObjectId('665453924260c60bfe3e83e5'),
      ObjectId('665453924260c60bfe3e83e6'),
      ObjectId('665453924260c60bfe3e83e7'),
      ObjectId('665453924260c60bfe3e83e8'),
      ObjectId('665453924260c60bfe3e83e9'),
      ObjectId('665453924260c60bfe3e83ea'),
      ObjectId('665453924260c60bfe3e83eb'),
      ObjectId('665453924260c60bfe3e83ec'),
      ObjectId('665453924260c60bfe3e83ed'),
      ObjectId('665453924260c60bfe3e83ee'),
      ObjectId('665453924260c60bfe3e83ef'),
      ObjectId('665453924260c60bfe3e83f0'),
      ObjectId('665453924260c60bfe3e83f1'),
      ObjectId('665453924260c60bfe3e83f2'),
      ObjectId('665453924260c60bfe3e83f3'),
      ObjectId('665453924260c60bfe3e83f4'),
      ObjectId('665453924260c60bfe3e83f5'),
      ObjectId('665453924260c60bfe3e83f6'),
      ObjectId('665453924260c60bfe3e83f7'),
      ObjectId('665453924260c60bfe3e83f8'),
      ObjectId('665453924260c60bfe3e83f9'),
      ObjectId('665453924260c60bfe3e83fa'),
      ObjectId('665453924260c60bfe3e83fb'),
      ObjectId('665453924260c60bfe3e83fc'),
      ObjectId('665453924260c60bfe3e83fd'),
      ObjectId('665453924260c60bfe3e83fe'),
      ObjectId('665453924260c60bfe3e83ff'),
      ObjectId('665453924260c60bfe3e8400'),
      ObjectId('665453924260c60bfe3e8401'),
      ObjectId('665453924260c60bfe3e8402'),
      ObjectId('665453924260c60bfe3e8403'),
      ObjectId('665453924260c60bfe3e8404'),
      ObjectId('665453924260c60bfe3e8405'),
      ObjectId('665453924260c60bfe3e8406'),
      ObjectId('665453924260c60bfe3e8407'),
      ObjectId('665453924260c60bfe3e8408'),
      ObjectId('665453924260c60bfe3e8409'),
      ObjectId('665453924260c60bfe3e840a'),
      ObjectId('665453924260c60bfe3e840b'),
      ObjectId('665453924260c60bfe3e840c'),
      ObjectId('665453924260c60bfe3e840d'),
      ObjectId('665453924260c60bfe3e840e'),
      ObjectId('665453924260c60bfe3e840f'),
      ObjectId('665453924260c60bfe3e8410'),
      ObjectId('665453924260c60bfe3e8411'),
      ObjectId('665453924260c60bfe3e8412'),
      ObjectId('665453924260c60bfe3e8413'),
      ObjectId('665453924260c60bfe3e8414'),
      ObjectId('665453924260c60bfe3e8415'),
      ObjectId('665453924260c60bfe3e8416'),
      ObjectId('665453924260c60bfe3e8417'),
      ObjectId('665453924260c60bfe3e8418'),
      ObjectId('665453924260c60bfe3e8419'),
      ObjectId('665453924260c60bfe3e841a'),
      ObjectId('665453924260c60bfe3e841b'),
      ObjectId('665453924260c60bfe3e841c'),
      ObjectId('665453924260c60bfe3e841d'),
      ObjectId('665453924260c60bfe3e841e'),
      ObjectId('665453924260c60bfe3e841f'),
      ObjectId('665453924260c60bfe3e8420'),
      ObjectId('665453924260c60bfe3e8421'),
      ObjectId('665453924260c60bfe3e8422'),
      ObjectId('665453924260c60bfe3e8423'),
      ObjectId('665453924260c60bfe3e8424'),
      ObjectId('665453924260c60bfe3e8425'),
      ObjectId('665453924260c60bfe3e8426'),
      ObjectId('665453924260c60bfe3e8427'),
      ObjectId('665453924260c60bfe3e8428'),
      ObjectId('665453924260c60bfe3e8429'),
      ObjectId('665453924260c60bfe3e842a'),
      ObjectId('665453924260c60bfe3e842b'),
      ObjectId('665453924260c60bfe3e842c'),
      ObjectId('665453924260c60bfe3e842d'),
      ObjectId('665453924260c60bfe3e842e'),
      ObjectId('665453924260c60bfe3e842f'),
      ObjectId('665453924260c60bfe3e8430'),
      ObjectId('665453924260c60bfe3e8431'),
      ObjectId('665453924260c60bfe3e8432'),
      ObjectId('665453924260c60bfe3e8433'),
      ObjectId('665453924260c60bfe3e8434'),
      ObjectId('665453924260c60bfe3e8435'),
      ObjectId('665453924260c60bfe3e8436'),
      ObjectId('665453924260c60bfe3e8437'),
      ObjectId('665453924260c60bfe3e8438'),
      ObjectId('665453924260c60bfe3e8439'),
      ObjectId('665453924260c60bfe3e843a'),
      ObjectId('665453924260c60bfe3e843b'),
      ObjectId('665453924260c60bfe3e843c'),
      ObjectId('665453924260c60bfe3e843d'),
      ObjectId('665453924260c60bfe3e843e'),
      ObjectId('665453924260c60bfe3e843f'),
      ObjectId('665453924260c60bfe3e8440'),
      ObjectId('665453924260c60bfe3e8441'),
      ObjectId('665453924260c60bfe3e8442'),
      ObjectId('665453924260c60bfe3e8443'),
      ObjectId('665453924260c60bfe3e8444'),
      ObjectId('665453924260c60bfe3e8445'),
      ObjectId('665453924260c60bfe3e8446'),
      ObjectId('665453924260c60bfe3e8447'),
      ObjectId('665453924260c60bfe3e8448'),
      ObjectId('665453924260c60bfe3e8449'),
      ObjectId('665453924260c60bfe3e844a'),
      ObjectId('665453924260c60bfe3e844b'),
      ObjectId('665453924260c60bfe3e844c'),
      ObjectId('665453924260c60bfe3e844d'),
      ObjectId('665453924260c60bfe3e844e'),
      ObjectId('665453924260c60bfe3e844f'),
      ObjectId('665453924260c60bfe3e8450'),
      ObjectId('665453924260c60bfe3e8451'),
      ObjectId('665453924260c60bfe3e8452'),
      ObjectId('665453924260c60bfe3e8453'),
      ObjectId('665453924260c60bfe3e8454'),
      ObjectId('665453924260c60bfe3e8455'),
      ObjectId('665453924260c60bfe3e8456'),
      ObjectId('665453924260c60bfe3e8457'),
      ObjectId('665453924260c60bfe3e8458'),
      ObjectId('665453924260c60bfe3e8459'),
      ObjectId('665453924260c60bfe3e845a'),
      ObjectId('665453924260c60bfe3e845b'),
      ObjectId('665453924260c60bfe3e845c'),
      ObjectId('665453924260c60bfe3e845d'),
      ObjectId('665453924260c60bfe3e845e'),
      ObjectId('665453924260c60bfe3e845f'),
      ObjectId('665453924260c60bfe3e8460'),
      ObjectId('665453924260c60bfe3e8461'),
      ObjectId('665453924260c60bfe3e8462'),
      ObjectId('665453924260c60bfe3e8463'),
      ObjectId('665453924260c60bfe3e8464'),
      ObjectId('665453924260c60bfe3e8465'),
      ObjectId('665453924260c60bfe3e8466'),
      ObjectId('665453924260c60bfe3e8467'),
      ObjectId('665453924260c60bfe3e8468'),
      ObjectId('665453924260c60bfe3e8469'),
      ObjectId('665453924260c60bfe3e846a')],
     TaskWorkflow(database=\<superduperdb.base.datalayer.Datalayer object at 0x13ed87e90\>, G=\<networkx.classes.digraph.DiGraph object at 0x13db43190\>))
</pre>
</details>

We can verify that the images are correctly saved by retrieved a single record:

```python
r = db['images'].find_one().execute()
r
```

<details>
<summary>Outputs</summary>
<pre>
    Document(\{'img': \<PIL.PngImagePlugin.PngImageFile image mode=RGB size=500x338 at 0x13EE96590\>, '_fold': 'train', '_schema': 'AUTO:img=pil_image', '_id': ObjectId('665453924260c60bfe3e8084')\})
</pre>
</details>

The contents of the `Document` may be accessed by calling `.unpack()`. You can see that the images were saved and retrieved correctly.

```python
r.unpack()['img']
```

<details>
<summary>Outputs</summary>
<div>![](/listening/10_0.png)</div>
</details>

We now build a `torch` model for text-2-image similarity using the `clip` library. In order to 
save the outputs correctly in the system, we add the `tensor` datatype to the model:

```python
import clip
import torch
from superduperdb.ext.torch import TorchModel, tensor


model, preprocess = clip.load("ViT-B/32", "cpu")

class ImageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model

    def forward(self, image_tensors):
        return self.model.encode_image(image_tensors)


dt = tensor(dtype='float', shape=(512,))


image_model = TorchModel(
    identifier='clip_image',
    object=ImageModel(),
    preprocess=preprocess,
    datatype=dt,
    loader_kwargs={'batch_size': 5},
)
```

<details>
<summary>Outputs</summary>

</details>

We can verify that this model gives us the correct outputs on the added data with the `.predict_one` method:

```python
image_model.predict_one(data[0]['img'])
```

<details>
<summary>Outputs</summary>
<pre>
    tensor([ 1.1590e-01, -1.1163e-01,  1.6210e-01,  3.2125e-01,  3.8310e-02,
            -2.9220e-01, -1.6120e-01, -1.0013e+00,  4.9538e-01,  1.4114e-01,
             7.6920e-02,  6.7780e-02, -1.0894e-01,  1.2793e-01, -2.6868e-01,
             4.6839e-01,  5.3715e-01,  7.9151e-02,  2.9155e-02,  2.5880e-01,
            -3.8380e-01,  8.6311e-02,  1.8946e-01,  1.6239e-01, -6.7896e-01,
             6.5299e-02,  4.9489e-01,  1.5839e-01,  8.3728e-02,  5.5632e-02,
            -1.0379e-01,  1.6675e-02, -3.3331e-01,  1.2236e-01,  6.2966e-01,
            -2.9543e-01,  6.5257e-01, -6.9910e-02,  2.0470e-01,  1.9606e+00,
            -1.2133e-01, -5.8945e-02, -1.3498e-01, -1.3249e-01,  2.0738e-01,
            -7.0674e-01,  1.3906e-01,  1.7988e-01,  7.0238e-02, -3.3584e-01,
            -2.3665e-01,  7.9334e-02, -1.0090e-01, -9.1650e-02,  1.7352e-01,
             1.1216e-01,  1.9300e-01,  4.8928e-01, -1.1548e-01,  5.7670e-02,
             6.2232e-01, -3.1829e-01, -2.3148e-01,  2.0030e-01, -4.0209e-02,
            -1.4554e-01,  4.4466e-01,  4.1464e-01, -2.4179e-01, -1.2451e-01,
             4.8724e-01, -4.7054e-02,  2.0541e-02, -4.9354e-03,  8.7666e-02,
             1.9785e-02, -1.6447e-01, -8.3285e-01, -1.2699e-01,  1.2998e-01,
            -2.3226e-01, -8.8556e-01, -1.7331e-01, -7.3583e-01,  1.2145e-01,
            -9.4184e-02,  9.7923e-01, -4.3598e-01,  3.9887e-01,  1.7900e-01,
             4.4396e-02, -4.5832e-01, -6.0410e+00, -4.3534e-01, -5.9481e-01,
             4.9649e-02, -2.7357e-02, -3.8371e-01, -9.3306e-01,  9.2320e-01,
            -1.9619e-01, -3.2803e-02, -1.9339e-02, -8.1225e-02,  4.7819e-01,
             3.1278e-02,  1.9959e-01, -2.0033e-01, -1.1671e-01, -3.9358e-01,
            -1.2317e-01, -9.5839e-01,  5.1610e-01,  1.2667e-01,  5.3419e-02,
            -8.2035e-02,  2.4216e-01,  7.0817e-03,  1.1345e-01,  5.7588e-02,
             3.5402e-01, -4.0607e-01, -4.1502e-02, -4.2697e-01,  3.4891e-01,
            -5.3146e-01, -1.0416e-01,  2.7876e-01,  1.4609e-01, -9.3855e-02,
            -1.9221e-01, -1.3139e-01, -1.7017e-01,  7.5767e-01, -4.1235e-01,
             1.9602e-01, -4.5141e-01, -2.7854e-01,  3.4447e-01,  2.3144e-02,
             5.0284e-01,  7.0880e-02, -4.6520e-02,  4.0040e-01,  1.0449e-01,
             5.3388e-01,  4.8889e-01, -1.3331e-01,  3.1072e-01,  1.7776e-01,
            -1.4810e-01,  3.3747e-01, -5.3392e-01, -1.0897e-01, -1.0275e-01,
            -1.0211e-01, -2.4024e-01, -9.3492e-02,  1.4864e-01, -5.1577e-01,
             5.0001e-01,  3.8927e-01, -2.4130e-01, -2.1834e-01,  2.5411e-01,
            -3.3644e-02, -2.9853e-01,  2.5758e-01, -1.6819e-01,  3.2913e-02,
            -2.6224e-01,  7.2353e-02, -1.1571e-01,  1.2307e-01,  5.3886e-02,
            -3.0080e-01,  7.7794e-01,  3.1006e-01,  1.4186e-01,  3.2553e-01,
             1.8123e-01,  5.2372e-02, -3.1844e-01,  9.4648e-03,  2.7061e-01,
             1.3351e-02,  1.5009e-01,  3.6278e-02,  3.6682e-01,  3.1048e-02,
             1.9435e-01, -5.7764e-02,  7.7549e-01, -1.4513e-01,  2.3016e-01,
            -3.9774e-02, -4.3843e-01, -2.7769e-01, -1.1522e+00, -7.6351e-02,
            -1.1522e-01, -1.1125e-01,  1.3012e-01,  3.2185e-01, -4.1987e-01,
             1.1690e-01,  2.2993e-01, -5.4273e-02,  7.1026e-02,  6.8535e-02,
            -3.1000e-01,  7.4613e-01, -1.7190e-01,  6.3011e-01, -1.7141e-02,
            -3.8054e-01, -2.2758e-01, -1.2865e-01,  9.5331e-01, -1.4831e-01,
             4.1659e-01,  3.2765e-01, -1.7633e-01, -4.9092e-02, -5.7158e-01,
             1.1250e-01,  8.9816e-02, -1.1677e-01,  3.0087e-01, -2.5121e-01,
             2.1989e-01, -4.8331e-01, -5.2032e-01,  8.8859e-02,  1.5607e-01,
             4.7345e-01, -2.8604e-01,  6.8347e-01,  3.0517e-02, -2.5008e-01,
             2.9491e-01, -2.0136e-01, -1.6408e-01,  7.5016e-02,  1.4922e-01,
             7.9002e-01,  3.8315e-02, -7.2039e-01, -2.8794e-01,  2.5925e-01,
            -4.6311e-01, -8.0276e-02, -2.5208e-01, -1.8260e-01,  1.0297e-01,
            -4.0524e-02, -4.5251e-04, -8.3430e-02,  4.3200e-01,  3.4515e-01,
            -5.0279e-01,  9.2067e-02, -2.5708e-01, -1.5032e+00, -3.8652e-02,
            -4.5196e-01,  8.5718e-03, -2.4990e-01, -3.9936e-02,  1.2828e-02,
            -3.1357e-01,  1.5526e-01,  5.4366e-01,  1.0955e-01, -5.4420e-01,
            -8.8090e-02, -2.4157e-01, -1.6538e-01, -3.1087e-01,  5.2663e-02,
             2.5429e-01, -4.8138e-02, -4.8485e-01,  3.8829e-01, -1.6188e-01,
            -1.2533e-01, -3.6505e-01, -1.3971e-01, -3.6999e-01, -8.9621e-02,
             1.0524e-01,  4.7519e-02,  1.4970e-01, -3.3221e-02,  5.3704e-02,
             7.2312e-02,  4.6348e-01, -1.5002e-02,  4.2172e-01,  2.0594e-01,
             1.4226e-01,  3.5545e-01,  2.2918e-02, -2.2476e-01, -3.0733e-03,
            -3.2779e-01, -5.9857e-02,  2.0830e-01,  5.2765e-01,  5.2056e-02,
            -3.0684e-01,  1.7120e-01,  7.5580e-01,  2.8488e-01, -1.0147e-01,
             2.9288e-01,  2.9232e-01,  5.2558e-01,  2.6853e-01, -6.0759e-01,
             3.1820e-01,  8.0559e-01, -2.9796e-01, -2.1610e-01,  3.1361e-02,
             3.7303e-02, -6.2717e-02,  5.5843e-02,  8.4793e-02, -2.6605e-01,
             9.1591e-02, -3.2060e-01, -3.1954e-01, -2.4636e-01,  4.0276e-01,
             4.6442e-01, -3.3635e-01, -1.8236e-01, -3.7923e-02,  2.3812e-01,
             2.5474e-02, -3.7982e-01, -3.2390e-01,  4.8698e-01,  4.0739e-01,
            -2.6965e-01,  3.4426e-02,  7.5702e-02,  3.2427e-01, -7.2199e-02,
            -2.7115e-01, -2.5527e-01,  5.9944e-01, -1.5569e-01, -9.0987e-02,
             4.4571e-01, -1.8296e-01, -4.9801e-01,  1.4001e-01,  6.0739e-01,
            -2.1250e-01, -3.2282e-02,  6.0238e-01,  1.1127e-01, -1.2075e+00,
             1.1658e-01, -1.8768e-01, -1.4897e-01,  4.3965e-01, -4.7514e-02,
            -1.1361e-01,  3.4682e-01,  7.2601e-02,  2.5298e-01, -5.5959e-01,
            -2.2657e-01,  1.0485e+00,  5.7782e-01, -6.8249e-01,  2.0709e-01,
             3.0597e-01, -4.3153e-01, -2.5387e-01,  3.8834e-01,  5.1551e-01,
             5.2100e-03,  3.3520e-01, -6.7791e-01,  6.6559e-01, -1.2692e+00,
            -1.4179e-01, -2.5137e-01,  2.1868e-01, -4.4603e-01,  8.2985e-02,
             3.0630e-01,  1.5204e-01, -1.2356e-01, -4.1893e-01, -6.5163e-01,
            -1.3898e-01, -3.1360e-01,  3.4516e-01,  4.4857e-01, -3.9492e-01,
             7.7293e-02, -4.2672e-02, -6.2462e-01,  7.2953e-01, -2.3767e-01,
             5.9171e-01, -1.5064e-01, -3.6149e-01,  2.1571e-01, -6.6212e-02,
            -3.1349e-01,  2.2127e-01, -1.1634e-01,  5.9028e-02, -5.0800e-01,
            -2.1432e-01,  1.0446e+00,  8.5178e-02,  2.9323e-01,  2.8574e-01,
             3.2103e-01, -2.4577e-01,  2.7432e-01,  1.1078e+00, -8.9108e-02,
             1.6229e-01,  1.6318e-01,  1.3020e-01, -1.5229e-02,  2.8989e-01,
            -5.8030e-01, -8.8477e-02,  4.3783e-01,  2.3567e-01,  1.1670e-01,
            -2.0998e-01,  2.5156e-01, -2.3052e-01, -2.1780e-01,  6.5135e-01,
            -1.6850e-01,  7.4846e-02, -1.8685e-01,  1.2253e-01, -4.1852e-01,
            -5.9780e-02, -7.0781e-01, -6.5484e-01,  7.1986e-01, -1.1047e-01,
             1.1983e-01,  5.5121e-01,  3.3051e-01,  4.2507e-01,  2.3734e-01,
             8.2319e-02, -2.0200e-01,  5.3302e-02,  5.6774e-01,  8.5539e-02,
             3.8295e-01,  1.8800e-02, -9.5386e-01,  7.4490e-02,  2.6489e-01,
             5.4485e-01, -1.8950e-01, -3.4931e-01, -8.4142e-02, -5.1462e-01,
            -5.5838e-02,  3.9012e-02,  2.8505e-01,  4.5606e-01, -2.2018e-01,
             1.4774e-01,  8.0736e-02, -6.9549e-02,  3.5784e-01, -2.9284e-01,
            -6.1763e-02, -5.6776e-02, -3.8342e-01, -3.8281e-01,  1.0981e-01,
             3.9572e-01, -3.6426e-01,  1.0481e-01, -2.7062e-01,  1.5884e-01,
             3.8384e-02,  1.0798e-01,  3.0277e-01,  2.1358e-01,  6.2604e-02,
            -3.6589e-01,  6.0148e-02, -2.0930e-01,  3.1359e-01,  7.2983e-01,
            -1.5445e-01, -5.2086e-02, -1.5287e-01,  3.1047e-01,  1.3831e+00,
             3.4753e-01,  2.2045e-01])
</pre>
</details>

Now we'd like to set up this model to compute outputs on the `'img'` key of each record. 
To do that we create a `Listener` (see [here](../apply_api/listener) for more information) which 
"listens" for incoming and existing data, and computes outputs on that data.

When new data is inserted, the model automatically will create outputs on that data. This is a very handy 
feature for productionizing AI and ML, since a data deployment needs to be keep up-to-date as far as possible.

```python
listener = image_model.to_listener(
    select=db['images'].find(),
    key='img',
    identifier='image_predictions',
)

_ = db.apply(listener)
```

<details>
<summary>Outputs</summary>
<pre>
    2024-May-27 11:34:16.44| WARNING  | Duncans-MBP.fritz.box| superduperdb.backends.local.artifacts:82   | File /tmp/e1635b227a7f3787dc79524d812915c342701260 already exists
    2024-May-27 11:34:17.37| INFO     | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function method_job at 0x10cd4bec0\>

</pre>
<pre>
    999it [00:00, 107213.29it/s]
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:30\<00:00,  6.58it/s]

</pre>
<pre>
    2024-May-27 11:34:49.03| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:783  | Adding 999 model outputs to `db`
    2024-May-27 11:34:49.67| WARNING  | Duncans-MBP.fritz.box| superduperdb.backends.mongodb.query:254  | Some delete ids are not executed , hence halting execution Please note the partially executed operations wont trigger any `model/listeners` unless CDC is active.
    2024-May-27 11:34:49.76| SUCCESS  | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x13ecc76d0\>.  function:\<function method_job at 0x10cd4bec0\> future:5d4a6013-900c-4582-9680-4043e1407519

</pre>
</details>

We can verify that the outputs are correctly inserted into the documents with this query. 
The outputs are saved in the `listener.outputs` field:

```python
list(listener.outputs_select.limit(1).execute())[0][listener.outputs].unpack()
```

<details>
<summary>Outputs</summary>
<pre>
    /Users/dodo/SuperDuperDB/superduperdb/superduperdb/ext/torch/encoder.py:52: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:212.)
      return torch.from_numpy(array)

</pre>
<pre>
    tensor([ 1.1590e-01, -1.1163e-01,  1.6210e-01,  3.2125e-01,  3.8310e-02,
            -2.9220e-01, -1.6120e-01, -1.0013e+00,  4.9538e-01,  1.4114e-01,
             7.6920e-02,  6.7780e-02, -1.0894e-01,  1.2793e-01, -2.6868e-01,
             4.6839e-01,  5.3715e-01,  7.9151e-02,  2.9155e-02,  2.5880e-01,
            -3.8380e-01,  8.6311e-02,  1.8946e-01,  1.6239e-01, -6.7896e-01,
             6.5299e-02,  4.9489e-01,  1.5839e-01,  8.3728e-02,  5.5632e-02,
            -1.0379e-01,  1.6675e-02, -3.3331e-01,  1.2236e-01,  6.2966e-01,
            -2.9543e-01,  6.5257e-01, -6.9910e-02,  2.0470e-01,  1.9606e+00,
            -1.2133e-01, -5.8945e-02, -1.3498e-01, -1.3249e-01,  2.0738e-01,
            -7.0674e-01,  1.3906e-01,  1.7988e-01,  7.0238e-02, -3.3584e-01,
            -2.3665e-01,  7.9334e-02, -1.0090e-01, -9.1650e-02,  1.7352e-01,
             1.1216e-01,  1.9300e-01,  4.8928e-01, -1.1548e-01,  5.7670e-02,
             6.2232e-01, -3.1829e-01, -2.3148e-01,  2.0030e-01, -4.0209e-02,
            -1.4554e-01,  4.4466e-01,  4.1464e-01, -2.4179e-01, -1.2451e-01,
             4.8724e-01, -4.7054e-02,  2.0541e-02, -4.9354e-03,  8.7666e-02,
             1.9785e-02, -1.6447e-01, -8.3285e-01, -1.2699e-01,  1.2998e-01,
            -2.3226e-01, -8.8556e-01, -1.7331e-01, -7.3583e-01,  1.2145e-01,
            -9.4184e-02,  9.7923e-01, -4.3598e-01,  3.9887e-01,  1.7900e-01,
             4.4396e-02, -4.5832e-01, -6.0410e+00, -4.3534e-01, -5.9481e-01,
             4.9649e-02, -2.7357e-02, -3.8371e-01, -9.3306e-01,  9.2320e-01,
            -1.9619e-01, -3.2803e-02, -1.9339e-02, -8.1225e-02,  4.7819e-01,
             3.1278e-02,  1.9959e-01, -2.0033e-01, -1.1671e-01, -3.9358e-01,
            -1.2317e-01, -9.5839e-01,  5.1610e-01,  1.2667e-01,  5.3419e-02,
            -8.2035e-02,  2.4216e-01,  7.0817e-03,  1.1345e-01,  5.7588e-02,
             3.5402e-01, -4.0607e-01, -4.1502e-02, -4.2697e-01,  3.4891e-01,
            -5.3146e-01, -1.0416e-01,  2.7876e-01,  1.4609e-01, -9.3855e-02,
            -1.9221e-01, -1.3139e-01, -1.7017e-01,  7.5767e-01, -4.1235e-01,
             1.9602e-01, -4.5141e-01, -2.7854e-01,  3.4447e-01,  2.3144e-02,
             5.0284e-01,  7.0880e-02, -4.6520e-02,  4.0040e-01,  1.0449e-01,
             5.3388e-01,  4.8889e-01, -1.3331e-01,  3.1072e-01,  1.7776e-01,
            -1.4810e-01,  3.3747e-01, -5.3392e-01, -1.0897e-01, -1.0275e-01,
            -1.0211e-01, -2.4024e-01, -9.3492e-02,  1.4864e-01, -5.1577e-01,
             5.0001e-01,  3.8927e-01, -2.4130e-01, -2.1834e-01,  2.5411e-01,
            -3.3644e-02, -2.9853e-01,  2.5758e-01, -1.6819e-01,  3.2913e-02,
            -2.6224e-01,  7.2353e-02, -1.1571e-01,  1.2307e-01,  5.3886e-02,
            -3.0080e-01,  7.7794e-01,  3.1006e-01,  1.4186e-01,  3.2553e-01,
             1.8123e-01,  5.2372e-02, -3.1844e-01,  9.4648e-03,  2.7061e-01,
             1.3351e-02,  1.5009e-01,  3.6278e-02,  3.6682e-01,  3.1048e-02,
             1.9435e-01, -5.7764e-02,  7.7549e-01, -1.4513e-01,  2.3016e-01,
            -3.9774e-02, -4.3843e-01, -2.7769e-01, -1.1522e+00, -7.6351e-02,
            -1.1522e-01, -1.1125e-01,  1.3012e-01,  3.2185e-01, -4.1987e-01,
             1.1690e-01,  2.2993e-01, -5.4273e-02,  7.1026e-02,  6.8535e-02,
            -3.1000e-01,  7.4613e-01, -1.7190e-01,  6.3011e-01, -1.7141e-02,
            -3.8054e-01, -2.2758e-01, -1.2865e-01,  9.5331e-01, -1.4831e-01,
             4.1659e-01,  3.2765e-01, -1.7633e-01, -4.9092e-02, -5.7158e-01,
             1.1250e-01,  8.9816e-02, -1.1677e-01,  3.0087e-01, -2.5121e-01,
             2.1989e-01, -4.8331e-01, -5.2032e-01,  8.8859e-02,  1.5607e-01,
             4.7345e-01, -2.8604e-01,  6.8347e-01,  3.0517e-02, -2.5008e-01,
             2.9491e-01, -2.0136e-01, -1.6408e-01,  7.5016e-02,  1.4922e-01,
             7.9002e-01,  3.8315e-02, -7.2039e-01, -2.8794e-01,  2.5925e-01,
            -4.6311e-01, -8.0276e-02, -2.5208e-01, -1.8260e-01,  1.0297e-01,
            -4.0524e-02, -4.5251e-04, -8.3430e-02,  4.3200e-01,  3.4515e-01,
            -5.0279e-01,  9.2067e-02, -2.5708e-01, -1.5032e+00, -3.8652e-02,
            -4.5196e-01,  8.5718e-03, -2.4990e-01, -3.9936e-02,  1.2828e-02,
            -3.1357e-01,  1.5526e-01,  5.4366e-01,  1.0955e-01, -5.4420e-01,
            -8.8090e-02, -2.4157e-01, -1.6538e-01, -3.1087e-01,  5.2663e-02,
             2.5429e-01, -4.8138e-02, -4.8485e-01,  3.8829e-01, -1.6188e-01,
            -1.2533e-01, -3.6505e-01, -1.3971e-01, -3.6999e-01, -8.9621e-02,
             1.0524e-01,  4.7519e-02,  1.4970e-01, -3.3221e-02,  5.3704e-02,
             7.2312e-02,  4.6348e-01, -1.5002e-02,  4.2172e-01,  2.0594e-01,
             1.4226e-01,  3.5545e-01,  2.2918e-02, -2.2476e-01, -3.0733e-03,
            -3.2779e-01, -5.9857e-02,  2.0830e-01,  5.2765e-01,  5.2056e-02,
            -3.0684e-01,  1.7120e-01,  7.5580e-01,  2.8488e-01, -1.0147e-01,
             2.9288e-01,  2.9232e-01,  5.2558e-01,  2.6853e-01, -6.0759e-01,
             3.1820e-01,  8.0559e-01, -2.9796e-01, -2.1610e-01,  3.1361e-02,
             3.7303e-02, -6.2717e-02,  5.5843e-02,  8.4793e-02, -2.6605e-01,
             9.1591e-02, -3.2060e-01, -3.1954e-01, -2.4636e-01,  4.0276e-01,
             4.6442e-01, -3.3635e-01, -1.8236e-01, -3.7923e-02,  2.3812e-01,
             2.5474e-02, -3.7982e-01, -3.2390e-01,  4.8698e-01,  4.0739e-01,
            -2.6965e-01,  3.4426e-02,  7.5702e-02,  3.2427e-01, -7.2199e-02,
            -2.7115e-01, -2.5527e-01,  5.9944e-01, -1.5569e-01, -9.0987e-02,
             4.4571e-01, -1.8296e-01, -4.9801e-01,  1.4001e-01,  6.0739e-01,
            -2.1250e-01, -3.2282e-02,  6.0238e-01,  1.1127e-01, -1.2075e+00,
             1.1658e-01, -1.8768e-01, -1.4897e-01,  4.3965e-01, -4.7514e-02,
            -1.1361e-01,  3.4682e-01,  7.2601e-02,  2.5298e-01, -5.5959e-01,
            -2.2657e-01,  1.0485e+00,  5.7782e-01, -6.8249e-01,  2.0709e-01,
             3.0597e-01, -4.3153e-01, -2.5387e-01,  3.8834e-01,  5.1551e-01,
             5.2100e-03,  3.3520e-01, -6.7791e-01,  6.6559e-01, -1.2692e+00,
            -1.4179e-01, -2.5137e-01,  2.1868e-01, -4.4603e-01,  8.2985e-02,
             3.0630e-01,  1.5204e-01, -1.2356e-01, -4.1893e-01, -6.5163e-01,
            -1.3898e-01, -3.1360e-01,  3.4516e-01,  4.4857e-01, -3.9492e-01,
             7.7293e-02, -4.2672e-02, -6.2462e-01,  7.2953e-01, -2.3767e-01,
             5.9171e-01, -1.5064e-01, -3.6149e-01,  2.1571e-01, -6.6212e-02,
            -3.1349e-01,  2.2127e-01, -1.1634e-01,  5.9028e-02, -5.0800e-01,
            -2.1432e-01,  1.0446e+00,  8.5178e-02,  2.9323e-01,  2.8574e-01,
             3.2103e-01, -2.4577e-01,  2.7432e-01,  1.1078e+00, -8.9108e-02,
             1.6229e-01,  1.6318e-01,  1.3020e-01, -1.5229e-02,  2.8989e-01,
            -5.8030e-01, -8.8477e-02,  4.3783e-01,  2.3567e-01,  1.1670e-01,
            -2.0998e-01,  2.5156e-01, -2.3052e-01, -2.1780e-01,  6.5135e-01,
            -1.6850e-01,  7.4846e-02, -1.8685e-01,  1.2253e-01, -4.1852e-01,
            -5.9780e-02, -7.0781e-01, -6.5484e-01,  7.1986e-01, -1.1047e-01,
             1.1983e-01,  5.5121e-01,  3.3051e-01,  4.2507e-01,  2.3734e-01,
             8.2319e-02, -2.0200e-01,  5.3302e-02,  5.6774e-01,  8.5539e-02,
             3.8295e-01,  1.8800e-02, -9.5386e-01,  7.4490e-02,  2.6489e-01,
             5.4485e-01, -1.8950e-01, -3.4931e-01, -8.4142e-02, -5.1462e-01,
            -5.5838e-02,  3.9012e-02,  2.8505e-01,  4.5606e-01, -2.2018e-01,
             1.4774e-01,  8.0736e-02, -6.9549e-02,  3.5784e-01, -2.9284e-01,
            -6.1763e-02, -5.6776e-02, -3.8342e-01, -3.8281e-01,  1.0981e-01,
             3.9572e-01, -3.6426e-01,  1.0481e-01, -2.7062e-01,  1.5884e-01,
             3.8384e-02,  1.0798e-01,  3.0277e-01,  2.1358e-01,  6.2604e-02,
            -3.6589e-01,  6.0148e-02, -2.0930e-01,  3.1359e-01,  7.2983e-01,
            -1.5445e-01, -5.2086e-02, -1.5287e-01,  3.1047e-01,  1.3831e+00,
             3.4753e-01,  2.2045e-01])
</pre>
</details>

Downstream of this first model, we now can add another smaller model, to classify images with configurable terms. 
Since the dataset is concerned with cats and dogs we create 2 downstream models classifying the images in 2 different ways.

```python
from superduperdb import ObjectModel


class Comparer:
    def __init__(self, words, text_features):
        self.targets = {w: text_features[i] for i, w in enumerate(words)}
        self.lookup = list(self.targets.keys())
        self.matrix = torch.stack(list(self.targets.values()))

    def __call__(self, vector):
        best = (self.matrix @ vector).topk(1)[1].item()
        return self.lookup[best]


cats_vs_dogs = ObjectModel(
    'cats_vs_dogs',
    object=Comparer(['cat', 'dog'], model.encode_text(clip.tokenize(['cat', 'dog']))),
).to_listener(
    select=db['images'].find(),
    key=listener.outputs,
)

            
felines_vs_canines = ObjectModel(
    'felines_vs_canines',
    object=Comparer(['feline', 'canine'], model.encode_text(clip.tokenize(['feline', 'canine']))),
).to_listener(
    select=db['images'].find(),
    key=listener.outputs,
)


db.apply(cats_vs_dogs)
db.apply(felines_vs_canines)
```

<details>
<summary>Outputs</summary>
<pre>
    2024-May-27 11:34:50.07| INFO     | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function method_job at 0x10cd4bec0\>

</pre>
<pre>
    999it [00:00, 130533.01it/s]

</pre>
<pre>
    2024-May-27 11:34:50.63| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:783  | Adding 999 model outputs to `db`
    2024-May-27 11:34:51.22| WARNING  | Duncans-MBP.fritz.box| superduperdb.backends.mongodb.query:254  | Some delete ids are not executed , hence halting execution Please note the partially executed operations wont trigger any `model/listeners` unless CDC is active.
    2024-May-27 11:34:51.22| SUCCESS  | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x13ecc76d0\>.  function:\<function method_job at 0x10cd4bec0\> future:3c5da5ea-a4d2-4964-89ed-5691e0e80d4b
    2024-May-27 11:34:51.23| INFO     | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function method_job at 0x10cd4bec0\>

</pre>
<pre>
    999it [00:00, 125897.17it/s]

</pre>
<pre>
    2024-May-27 11:34:51.51| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:783  | Adding 999 model outputs to `db`
    2024-May-27 11:34:52.08| WARNING  | Duncans-MBP.fritz.box| superduperdb.backends.mongodb.query:254  | Some delete ids are not executed , hence halting execution Please note the partially executed operations wont trigger any `model/listeners` unless CDC is active.
    2024-May-27 11:34:52.08| SUCCESS  | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x13ecc76d0\>.  function:\<function method_job at 0x10cd4bec0\> future:f46f6471-7895-482d-ab03-ec7344235512

</pre>
<pre>
    ([\<superduperdb.jobs.job.ComponentJob at 0x142c03d90\>],
     Listener(identifier='component/listener/felines_vs_canines/e92d248d-8ea8-4e2b-8254-5988acfea072', uuid='e92d248d-8ea8-4e2b-8254-5988acfea072', key='_outputs.b55cc5c5-9427-40ee-95a3-227e659cd783', model=ObjectModel(identifier='felines_vs_canines', uuid='65f2761b-9d4c-47fb-a5a9-f5e95c81bda9', signature='*args,**kwargs', datatype=None, output_schema=None, flatten=False, model_update_kwargs=\{\}, predict_kwargs=\{\}, compute_kwargs=\{\}, validation=None, metric_values=\{\}, num_workers=0, object=\<__main__.Comparer object at 0x177f42bd0\>), select=images.find(), active=True, predict_kwargs=\{\}))
</pre>
</details>

We can verify that both downstream models have written their outputs to the database by querying a document:

```python
r = db['images'].find_one().execute()

print(r[cats_vs_dogs.outputs])
print(r[felines_vs_canines.outputs])
```

<details>
<summary>Outputs</summary>
<pre>
    cat
    feline

</pre>
</details>

Let's check that the predictions make sense for the inserted images:

```python
db['images'].find_one({cats_vs_dogs.outputs: 'cat'}).execute()['img']
```

<details>
<summary>Outputs</summary>
<div>![](/listening/24_0.png)</div>
</details>

```python
db['images'].find_one({felines_vs_canines.outputs: 'feline'}).execute()['img']
```

<details>
<summary>Outputs</summary>
<div>![](/listening/25_0.png)</div>
</details>

```python
db['images'].find_one({cats_vs_dogs.outputs: 'dog'}).execute()['img']
```

<details>
<summary>Outputs</summary>
<div>![](/listening/26_0.png)</div>
</details>

```python
db['images'].find_one({felines_vs_canines.outputs: 'canine'}).execute()['img']
```

<details>
<summary>Outputs</summary>
<div>![](/listening/27_0.png)</div>
</details>

Now that we have installed the models using `Listener`, we can insert new data. This 
data should be automatically processed by the installed models:

```python
db['images'].insert_one(Document({**data[-1], 'new': True})).execute()
```

<details>
<summary>Outputs</summary>
<pre>
    2024-May-27 11:36:59.83| WARNING  | Duncans-MBP.fritz.box| superduperdb.misc.annotations:117  | add is deprecated and will be removed in a future release.
    2024-May-27 11:37:00.62| INFO     | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function callable_job at 0x10cd4be20\>
    2024-May-27 11:37:00.62| SUCCESS  | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x13ecc76d0\>.  function:\<function callable_job at 0x10cd4be20\> future:ce572425-a6bd-4170-bad9-6ff56598388d
    2024-May-27 11:37:00.90| INFO     | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function method_job at 0x10cd4bec0\>

</pre>
<pre>
    1it [00:00, 305.06it/s]
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00\<00:00, 11.75it/s]

</pre>
<pre>
    2024-May-27 11:37:01.54| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:783  | Adding 1 model outputs to `db`
    2024-May-27 11:37:01.54| WARNING  | Duncans-MBP.fritz.box| superduperdb.backends.mongodb.query:254  | Some delete ids are not executed , hence halting execution Please note the partially executed operations wont trigger any `model/listeners` unless CDC is active.
    2024-May-27 11:37:01.54| SUCCESS  | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x13ecc76d0\>.  function:\<function method_job at 0x10cd4bec0\> future:189c7a61-61f3-4fbc-837c-4bdd38d6241d
    2024-May-27 11:37:01.54| INFO     | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function method_job at 0x10cd4bec0\>

</pre>
<pre>
    1it [00:00, 319.59it/s]

</pre>
<pre>
    2024-May-27 11:37:01.55| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:783  | Adding 1 model outputs to `db`
    2024-May-27 11:37:01.55| WARNING  | Duncans-MBP.fritz.box| superduperdb.backends.mongodb.query:254  | Some delete ids are not executed , hence halting execution Please note the partially executed operations wont trigger any `model/listeners` unless CDC is active.
    2024-May-27 11:37:01.55| SUCCESS  | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x13ecc76d0\>.  function:\<function method_job at 0x10cd4bec0\> future:1c603b81-4891-4d9b-8c1b-48039e77e27d
    2024-May-27 11:37:01.55| INFO     | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function method_job at 0x10cd4bec0\>

</pre>
<pre>
    1it [00:00, 322.39it/s]
</pre>
<pre>
    2024-May-27 11:37:01.56| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:783  | Adding 1 model outputs to `db`
    2024-May-27 11:37:01.56| WARNING  | Duncans-MBP.fritz.box| superduperdb.backends.mongodb.query:254  | Some delete ids are not executed , hence halting execution Please note the partially executed operations wont trigger any `model/listeners` unless CDC is active.
    2024-May-27 11:37:01.56| SUCCESS  | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x13ecc76d0\>.  function:\<function method_job at 0x10cd4bec0\> future:a5b04a66-feac-4270-8685-edb0de90142f

</pre>
<pre>
    

</pre>
<pre>
    ([ObjectId('6654543b4260c60bfe3e8479')],
     TaskWorkflow(database=\<superduperdb.base.datalayer.Datalayer object at 0x13ed87e90\>, G=\<networkx.classes.digraph.DiGraph object at 0x15c991d10\>))
</pre>
</details>

We can verify this by querying the data again:

```python
r = db['images'].find_one({'new': True}).execute().unpack()
r['img']
```

<details>
<summary>Outputs</summary>
<div>![](/listening/31_0.png)</div>
</details>

You see here that the models have been called in the correct order on the newly added data and the outputs saved 
to the new record:

```python
r['_outputs']
```

<details>
<summary>Outputs</summary>
<pre>
    \{'b55cc5c5-9427-40ee-95a3-227e659cd783': tensor([-1.9784e-01, -1.1985e-02,  1.5070e-02,  2.3934e-01,  9.1826e-02,
             -6.8618e-01,  5.9504e-01, -1.3494e+00, -1.2732e-01,  1.8213e-01,
             -2.4697e-01,  4.2079e-01,  5.8837e-03, -6.9505e-02,  3.9881e-01,
             -1.6329e-01,  1.0990e+00, -2.4527e-01,  4.1645e-01,  2.2653e-01,
             -4.2598e-01, -1.3891e-01,  3.8267e-01, -4.2262e-01,  4.2060e-02,
              1.8807e-01,  5.9752e-01,  2.3998e-01,  5.2914e-01,  6.6496e-02,
              2.9223e-01,  3.7576e-01, -2.6834e-01, -3.8394e-01, -5.1581e-01,
              6.0321e-02,  4.3297e-01, -9.6605e-02, -4.9369e-01,  1.7737e+00,
             -5.3931e-01, -3.9127e-01,  1.7854e-01, -2.3078e-01,  2.8036e-01,
              1.3256e-01,  1.7078e-01, -1.3245e-02, -1.1761e-01,  4.4366e-02,
             -2.0976e-01,  4.0344e-02, -3.3105e-02,  1.1045e-01,  4.9561e-01,
              1.8460e-02, -4.0324e-01,  2.0999e-01, -8.3989e-02, -4.0953e-01,
              6.7217e-01,  2.4515e-01,  7.3640e-01, -5.4399e-02, -3.6352e-01,
              3.7475e-02,  2.2933e-01,  2.1635e-01, -3.7907e-01,  7.3136e-02,
              1.7207e-01, -2.2663e-01,  2.0013e-01,  3.5541e-01, -3.6459e-01,
             -1.5265e-01, -2.8436e-01, -3.8413e-01, -2.1256e-02,  1.6117e-01,
              8.1823e-02, -2.8004e-01,  5.4281e-02, -6.5066e-01,  4.6435e-01,
             -3.6090e-01,  1.4236e+00, -3.5425e-01, -2.0861e-01,  1.5555e-01,
             -2.3906e-01, -2.6277e-01, -6.1501e+00,  6.8107e-01, -4.6798e-01,
             -2.5333e-01,  1.2230e-01,  8.8723e-02, -1.0091e+00,  1.0498e+00,
             -1.6454e-01, -1.4257e-01, -7.2294e-02, -1.9788e-01,  2.5033e-01,
              9.9614e-02,  1.6566e-01, -2.6232e-01, -2.2362e-01, -3.7903e-01,
             -2.1926e-01, -4.8382e-01,  4.1420e-01, -2.5579e-02,  4.6423e-02,
              3.9816e-01, -2.2495e-01, -2.0562e-01, -4.5065e-01,  1.1380e-01,
              5.0083e-01, -3.8125e-01,  1.2009e-01, -1.7179e-01, -9.2985e-02,
             -9.0663e-01,  1.1145e-01,  4.6361e-01, -4.8051e-01, -3.1573e-01,
              3.9665e-01,  8.7184e-02, -4.5003e-01,  7.2500e-01, -2.8902e-01,
              2.6280e-01,  1.4857e-01, -4.4423e-02, -3.3938e-01, -2.3859e-01,
              4.9975e-02, -2.1402e-01, -3.5357e-01,  1.9798e-01, -3.5788e-01,
              2.6473e-01,  3.6177e-01,  3.8417e-01, -1.6142e-01,  4.3199e-02,
             -3.2399e-01,  3.5889e-01, -6.3973e-01,  3.7262e-01, -8.1334e-01,
              2.0936e-01, -5.3070e-01,  1.2967e-01,  9.4323e-02,  1.4453e-01,
             -2.8726e-01,  3.4551e-01, -4.3649e-02,  2.1140e-01,  3.9445e-01,
             -3.2790e-01, -4.2833e-01,  2.5720e-01,  9.5053e-02,  4.3625e-01,
             -5.7717e-02,  2.2466e-01,  7.9365e-02,  4.0048e-02,  2.5109e-01,
              1.0167e-01,  1.4203e+00, -2.5085e-01,  2.8730e-01, -2.0506e-01,
             -6.9146e-02,  2.2863e-01,  5.7341e-01, -4.3942e-02, -3.0144e-01,
             -6.5863e-02, -3.5006e-01,  3.7668e-01, -2.0926e-01,  2.0152e-01,
             -2.4407e-03, -5.2490e-02,  2.6947e-01,  1.4438e-01,  2.8786e-01,
              2.7246e-01,  3.1032e-01, -2.0468e-01, -1.1268e+00, -5.5739e-01,
              4.1084e-02, -1.0877e-01,  5.1607e-01,  8.3844e-02, -3.7792e-01,
             -4.8277e-01,  1.9685e-01, -2.5837e-01, -3.4591e-01,  3.5694e-01,
             -4.3230e-01,  1.0324e-01, -4.7433e-01,  3.6803e-03,  9.4960e-02,
             -2.3161e-01, -3.2248e-01,  2.6306e-01,  1.1879e+00,  4.2923e-02,
              6.0710e-01,  3.5460e-01,  9.0147e-02,  2.4508e-01, -4.2532e-01,
             -2.7915e-01,  4.6251e-01, -3.4883e-01, -8.6505e-02,  4.5012e-01,
              3.1542e-01,  6.6807e-02, -4.2769e-01,  3.6456e-01, -2.1610e-01,
              6.7809e-01,  3.0854e-01, -1.8361e-01,  1.9379e-02,  5.7221e-02,
              2.3852e-01,  1.4775e-01, -5.2261e-02,  1.0130e-02, -1.5239e-01,
              8.6012e-01,  4.6111e-01, -6.5358e-01,  3.6930e-01,  4.3411e-01,
              4.9753e-01,  8.0125e-02,  2.4168e-01, -5.1495e-02,  3.2372e-01,
             -1.3840e-01, -4.4837e-01, -1.1076e-01,  1.4098e+00,  2.3750e-01,
              1.2766e-01,  2.1274e-01,  5.0297e-01, -1.1665e+00,  2.7810e-01,
             -3.8077e-01, -1.8598e-01, -3.1121e-01, -3.7083e-01,  2.4960e-01,
             -4.1929e-01, -2.4243e-01,  3.9122e-01,  1.7913e-02,  2.7623e-01,
              7.9261e-02, -1.8902e-02,  3.6915e-01,  2.3819e-01,  4.9772e-02,
             -3.8237e-01, -6.3968e-02, -7.0589e-01, -1.1346e-01, -3.8071e-01,
             -2.6041e-01,  7.2083e-01,  4.2882e-02, -2.5185e-01,  7.4913e-02,
              2.4498e-02, -1.2440e-01, -1.3673e-01,  4.0574e-01,  1.1170e-01,
             -1.4629e-01,  4.2372e-01,  5.7550e-01,  2.1063e-01,  2.1003e-01,
             -6.9390e-01,  1.2443e-01, -1.6181e-01,  4.6245e-02,  1.4564e-01,
             -1.9245e-01,  1.8174e-01, -8.2501e-02,  7.1926e-03,  4.4878e-01,
             -2.2032e-01,  6.9630e-01,  7.2316e-01, -3.4519e-01,  9.0801e-02,
              1.1218e-02, -2.1404e-02,  8.2681e-01, -3.0264e-01, -7.6556e-02,
              2.9048e-01,  3.4054e-01,  2.5881e-02,  1.9680e-01,  3.1750e-01,
              4.4904e-01, -5.4009e-02,  3.5515e-01, -2.5754e-01, -3.0279e-01,
             -3.3378e-01, -3.1998e-01,  3.8379e-01, -1.5646e-01,  2.8057e-02,
              1.3828e-01,  5.9443e-02,  1.5303e-01,  1.5804e-01, -2.3152e-01,
              1.8590e-01, -1.7906e-01, -4.8659e-02,  3.5145e-01,  5.8912e-01,
             -1.8931e-01,  1.1240e-01,  8.6221e-02, -1.2742e-01,  1.1433e-02,
             -1.0033e-01,  1.9416e-01,  7.5975e-01, -8.4944e-03, -1.0347e-01,
              3.3857e-01, -3.2450e-03, -2.5949e-01, -1.3049e-01, -2.7429e-01,
              3.6076e-01,  2.2456e-02,  1.8395e-01, -1.1796e-02, -1.1933e+00,
              1.9488e-01, -4.7571e-01,  8.2972e-02,  2.0567e-02, -5.6939e-01,
              1.8543e-01,  1.5819e-01, -2.6344e-02, -1.3586e-01, -3.0559e-01,
             -9.1858e-02,  1.5596e+00, -1.1252e-01, -9.1376e-01,  1.4403e-01,
              5.4933e-01, -8.1129e-03, -2.0602e-01,  2.7938e-01, -1.7516e-01,
              3.3268e-01, -7.0229e-01, -1.8966e-01,  3.7934e-01, -1.4382e+00,
              4.1491e-02, -7.4356e-01,  4.4283e-01,  2.2037e-01,  9.5358e-02,
              1.8080e-01, -1.0913e-01, -6.1602e-02, -4.7201e-01, -1.2602e-01,
             -2.5680e-01, -1.3158e-01, -9.7970e-02,  3.1354e-01,  2.4643e-01,
             -3.5605e-02,  2.2407e-01, -5.6102e-01, -4.3781e-02, -6.6948e-02,
              2.3365e-01,  2.0224e-01, -4.8406e-01, -1.8340e-01,  2.6515e-01,
              3.5039e-02,  1.4085e-01, -3.1442e-01,  3.6208e-01, -4.2783e-01,
              1.0023e-01,  5.7867e-01,  9.1382e-02,  7.7051e-01,  3.5109e-02,
              2.1105e-01,  1.7448e-01, -8.4044e-03,  6.7720e-01, -4.8796e-01,
             -2.3340e-01,  1.2858e-01, -7.6849e-02,  2.9516e-01,  6.7951e-02,
             -2.0271e-01, -4.4297e-01,  3.9987e-01,  2.8524e-01,  9.0605e-03,
              1.6964e-01, -2.3849e-01, -2.1865e-01, -5.8215e-02, -1.5098e-01,
              3.9566e-02, -1.3141e-02, -4.2397e-01, -7.0880e-02, -2.0458e-01,
              3.3506e-01, -2.1963e-01,  2.9304e-01,  3.5155e-01, -8.0974e-02,
             -2.9205e-02,  2.8120e-01,  7.4953e-01, -2.0307e-01,  2.9373e-01,
              1.0246e-01,  2.5897e-01,  6.0300e-03, -1.3105e-01,  3.3122e-01,
              3.9685e-01, -5.5882e-01,  2.2932e-01,  9.2667e-03, -2.8109e-01,
             -2.2059e-01, -1.4619e-02, -4.6908e-01, -4.6781e-01, -2.1942e-02,
              9.2600e-02, -4.1449e-01, -2.8908e-01, -3.4762e-01, -9.6450e-02,
              2.1795e-01,  2.1334e-01, -2.4707e-01, -1.4550e-01, -2.4918e-01,
              2.2423e-01, -7.1012e-02, -7.5370e-02, -2.2632e-01, -2.5647e-01,
             -8.1591e-01, -3.4810e-01,  4.4667e-01, -1.4343e-01,  4.2052e-01,
              7.8439e-02, -4.4730e-02,  3.6128e-01, -4.1957e-01, -1.6465e-01,
              5.5771e-02,  3.9537e-02, -1.5157e-01,  1.5093e-01, -3.7800e-01,
              9.5192e-02,  5.4860e-02, -1.2122e-01, -1.1731e-01,  1.5047e+00,
              4.1262e-01,  3.6463e-01]),
     'b53e76c0-9e5d-41a9-8c8b-014a13a61cfe': 'dog',
     'e92d248d-8ea8-4e2b-8254-5988acfea072': 'canine'\}
</pre>
</details>
