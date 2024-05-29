
# Basic RAG tutorial with templates

:::info
In this tutorial we show you how to do retrieval augmented generation (RAG) with `superduperdb`.
Note that this is just an example of the flexibility and power which `superduperdb` gives 
to developers. `superduperdb` is about much more than RAG and LLMs. 
:::

As in the vector-search tutorial we'll use `superduperdb` documentation for the tutorial.
We'll add this to a testing database by downloading the data snapshot:

```python
!curl -O https://superduperdb-public-demo.s3.amazonaws.com/text.json
import json

from superduperdb import superduper, Document

db = superduper('mongomock://test')

with open('text.json') as f:
    data = json.load(f)

_ = db['docu'].insert_many([{'txt': r} for r in data]).execute()
```

<details>
<summary>Outputs</summary>
<pre>
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  479k  100  479k    0     0   501k      0 --:--:-- --:--:-- --:--:--  504k
    2024-Jun-02 14:23:40.34| INFO     | Duncans-MBP.fritz.box| superduperdb.base.build:69   | Data Client is ready. mongomock.MongoClient('localhost', 27017)
    2024-Jun-02 14:23:40.35| INFO     | Duncans-MBP.fritz.box| superduperdb.base.build:42   | Connecting to Metadata Client with engine:  mongomock.MongoClient('localhost', 27017)
    2024-Jun-02 14:23:40.36| INFO     | Duncans-MBP.fritz.box| superduperdb.base.build:155  | Connecting to compute client: None
    2024-Jun-02 14:23:40.36| INFO     | Duncans-MBP.fritz.box| superduperdb.base.datalayer:85   | Building Data Layer
    2024-Jun-02 14:23:40.36| INFO     | Duncans-MBP.fritz.box| superduperdb.base.build:220  | Configuration: 
     +---------------+------------------+
    | Configuration |      Value       |
    +---------------+------------------+
    |  Data Backend | mongomock://test |
    +---------------+------------------+
    2024-Jun-02 14:23:40.37| INFO     | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function callable_job at 0x11e98dda0\>
    2024-Jun-02 14:23:40.38| SUCCESS  | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x2a8ed7050\>.  function:\<function callable_job at 0x11e98dda0\> future:ebe43b87-1388-4247-8502-ed2da8659ecd

</pre>
</details>

Let's verify the data in the `db` by querying one datapoint:

```python
db['docu'].find_one().execute()
```

<details>
<summary>Outputs</summary>
<pre>
    Document(\{'txt': "---\nsidebar_position: 5\n---\n\n# Encoding data\n\nIn AI, typical types of data are:\n\n- **Numbers** (integers, floats, etc.)\n- **Text**\n- **Images**\n- **Audio**\n- **Videos**\n- **...bespoke in house data**\n\nMost databases don't support any data other than numbers and text.\nSuperDuperDB enables the use of these more interesting data-types using the `Document` wrapper.\n\n### `Document`\n\nThe `Document` wrapper, wraps dictionaries, and is the container which is used whenever \ndata is exchanged with your database. That means inputs, and queries, wrap dictionaries \nused with `Document` and also results are returned wrapped with `Document`.\n\nWhenever the `Document` contains data which is in need of specialized serialization,\nthen the `Document` instance contains calls to `DataType` instances.\n\n### `DataType`\n\nThe [`DataType` class](../apply_api/datatype), allows users to create and encoder custom datatypes, by providing \ntheir own encoder/decoder pairs.\n\nHere is an example of applying an `DataType` to add an image to a `Document`:\n\n```python\nimport pickle\nimport PIL.Image\nfrom superduperdb import DataType, Document\n\nimage = PIL.Image.open('my_image.jpg')\n\nmy_image_encoder = DataType(\n    identifier='my-pil',\n    encoder=lambda x: pickle.dumps(x),\n    decoder=lambda x: pickle.loads(x),\n)\n\ndocument = Document(\{'img': my_image_encoder(image)\})\n```\n\nThe bare-bones dictionary may be exposed with `.unpack()`:\n\n```python\n\>\>\> document.unpack()\n\{'img': \<PIL.PngImagePlugin.PngImageFile image mode=P size=400x300\>\}\n```\n\nBy default, data encoded with `DataType` is saved in the database, but developers \nmay alternatively save data in the `db.artifact_store` instead. \n\nThis may be achiever by specifying the `encodable=...` parameter:\n\n```python\nmy_image_encoder = DataType(\n    identifier='my-pil',\n    encoder=lambda x: pickle.dumps(x),\n    decoder=lambda x: pickle.loads(x),\n    encodable='artifact',    # saves to disk/ db.artifact_store\n    # encodable='lazy_artifact', # Just in time loading\n)\n```\n\nThe `encodable` specifies the type of the output of the `__call__` method, \nwhich will be a subclass of `superduperdb.components.datatype._BaseEncodable`.\nThese encodables become leaves in the tree defines by a `Document`.\n\n### `Schema`\n\nA `Schema` allows developers to connect named fields of dictionaries \nor columns of `pandas.DataFrame` objects with `DataType` instances.\n\nA `Schema` is used, in particular, for SQL databases/ tables, and for \nmodels that return multiple outputs.\n\nHere is an example `Schema`, which is used together with text and image \nfields:\n\n```python\ns = Schema('my-schema', fields=\{'my-text': 'str', 'my-image': my_image_encoder\})\n```\n", '_fold': 'train', '_id': ObjectId('665c644c53dcb972da5a9928')\})
</pre>
</details>

The first step in a RAG application is to create a `VectorIndex`. The results of searching 
with this index will be used as input to the LLM for answering questions.

Read about `VectorIndex` [here](../apply_api/vector_index.md) and follow along the tutorial on 
vector-search [here](./vector_search.md).

```python
import requests 

from superduperdb import Stack, Document, VectorIndex, Listener, vector
from superduperdb.ext.sentence_transformers.model import SentenceTransformer
from superduperdb.base.code import Code

def postprocess(x):
    return x.tolist()

datatype = vector(shape=384, identifier="my-vec")
    
model = SentenceTransformer(
    identifier="my-embedding",
    datatype=datatype,
    predict_kwargs={"show_progress_bar": True},
    signature="*args,**kwargs",
    model="all-MiniLM-L6-v2",      
    device="cpu",
    postprocess=Code.from_object(postprocess),
)

listener = Listener(
    identifier="my-listener",
    model=model,
    key='txt',
    select=db['docu'].find(),
    predict_kwargs={'max_chunk_size': 50},
)

vector_index = VectorIndex(
    identifier="my-index",
    indexing_listener=listener,
    measure="cosine"
)

db.apply(vector_index)
```

<details>
<summary>Outputs</summary>
<pre>
    from superduperdb import code
    
    @code
    def postprocess(x):
        return x.tolist()
    

</pre>
<pre>
    /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(

</pre>
<pre>
    2024-Jun-02 14:23:58.41| INFO     | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function method_job at 0x11e98de40\>

</pre>
<pre>
    204it [00:00, 149744.14it/s]
</pre>
<pre>
    2024-Jun-02 14:23:59.54| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:752  | Computing chunk 0/4

</pre>
<pre>
    

</pre>
<pre>
    Batches:   0%|          | 0/2 [00:00\<?, ?it/s]
</pre>
<pre>
    2024-Jun-02 14:24:00.55| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:776  | Adding 50 model outputs to `db`
    2024-Jun-02 14:24:00.58| WARNING  | Duncans-MBP.fritz.box| superduperdb.backends.mongodb.query:316  | Some delete ids are not executed , hence halting execution Please note the partially executed operations wont trigger any `model/listeners` unless CDC is active.
    2024-Jun-02 14:24:00.58| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:752  | Computing chunk 1/4

</pre>
<pre>
    Batches:   0%|          | 0/2 [00:00\<?, ?it/s]
</pre>
<pre>
    2024-Jun-02 14:24:01.40| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:776  | Adding 50 model outputs to `db`
    2024-Jun-02 14:24:01.43| WARNING  | Duncans-MBP.fritz.box| superduperdb.backends.mongodb.query:316  | Some delete ids are not executed , hence halting execution Please note the partially executed operations wont trigger any `model/listeners` unless CDC is active.
    2024-Jun-02 14:24:01.43| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:752  | Computing chunk 2/4

</pre>
<pre>
    Batches:   0%|          | 0/2 [00:00\<?, ?it/s]
</pre>
<pre>
    2024-Jun-02 14:24:02.28| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:776  | Adding 50 model outputs to `db`
    2024-Jun-02 14:24:02.30| WARNING  | Duncans-MBP.fritz.box| superduperdb.backends.mongodb.query:316  | Some delete ids are not executed , hence halting execution Please note the partially executed operations wont trigger any `model/listeners` unless CDC is active.
    2024-Jun-02 14:24:02.30| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:752  | Computing chunk 3/4

</pre>
<pre>
    Batches:   0%|          | 0/2 [00:00\<?, ?it/s]
</pre>
<pre>
    2024-Jun-02 14:24:03.13| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:776  | Adding 50 model outputs to `db`
    2024-Jun-02 14:24:03.16| WARNING  | Duncans-MBP.fritz.box| superduperdb.backends.mongodb.query:316  | Some delete ids are not executed , hence halting execution Please note the partially executed operations wont trigger any `model/listeners` unless CDC is active.
    2024-Jun-02 14:24:03.16| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:752  | Computing chunk 4/4

</pre>
<pre>
    Batches:   0%|          | 0/1 [00:00\<?, ?it/s]
</pre>
<pre>
    2024-Jun-02 14:24:03.26| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:776  | Adding 4 model outputs to `db`
    2024-Jun-02 14:24:03.26| WARNING  | Duncans-MBP.fritz.box| superduperdb.backends.mongodb.query:316  | Some delete ids are not executed , hence halting execution Please note the partially executed operations wont trigger any `model/listeners` unless CDC is active.
    2024-Jun-02 14:24:03.26| SUCCESS  | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x2a8ed7050\>.  function:\<function method_job at 0x11e98de40\> future:ac399012-8213-481a-b537-3d187fb69583
    2024-Jun-02 14:24:03.26| INFO     | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function callable_job at 0x11e98dda0\>
    2024-Jun-02 14:24:04.56| INFO     | Duncans-MBP.fritz.box| superduperdb.base.datalayer:169  | Loading vectors of vector-index: 'my-index'
    2024-Jun-02 14:24:04.56| INFO     | Duncans-MBP.fritz.box| superduperdb.base.datalayer:179  | docu.find(documents[0], documents[1])

</pre>
<pre>
    Loading vectors into vector-table...: 204it [00:00, 3031.62it/s]
</pre>
<pre>
    2024-Jun-02 14:24:04.63| SUCCESS  | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x2a8ed7050\>.  function:\<function callable_job at 0x11e98dda0\> future:41ae1218-3899-4588-bef6-481acee98e25

</pre>
<pre>
    

</pre>
<pre>
    ([\<superduperdb.jobs.job.ComponentJob at 0x2dfe3a810\>,
      \<superduperdb.jobs.job.FunctionJob at 0x2acd55dd0\>],
     VectorIndex(identifier='my-index', uuid='7cb9de9f-4cc8-4944-a297-f6a433c51d19', indexing_listener=Listener(identifier='my-listener', uuid='81ea6d64-21f0-4552-b234-1bcf8094c35f', key='txt', model=SentenceTransformer(preferred_devices=('cuda', 'mps', 'cpu'), device='cpu', identifier='my-embedding', uuid='db4daee6-22fe-43fe-8a57-97ced878ef2a', signature='*args,**kwargs', datatype=DataType(identifier='my-vec', uuid='dbdb8706-10f7-4377-952b-b83b81c6624a', encoder=None, decoder=None, info=None, shape=(384,), directory=None, encodable='native', bytes_encoding=\<BytesEncoding.BYTES: 'Bytes'\>, intermediate_type='bytes', media_type=None), output_schema=None, flatten=False, model_update_kwargs=\{\}, predict_kwargs=\{'show_progress_bar': True\}, compute_kwargs=\{\}, validation=None, metric_values=\{\}, object=SentenceTransformer(
       (0): Transformer(\{'max_seq_length': 256, 'do_lower_case': False\}) with Transformer model: BertModel 
       (1): Pooling(\{'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True\})
       (2): Normalize()
     ), model='all-MiniLM-L6-v2', preprocess=None, postprocess=Code(identifier='', uuid='cb1d9759-5061-4f39-a91d-e25b959e2a18', code='from superduperdb import code\n\n@code\ndef postprocess(x):\n    return x.tolist()\n')), select=docu.find(), active=True, predict_kwargs=\{'max_chunk_size': 50\}), compatible_listener=None, measure='cosine', metric_values=\{\}))
</pre>
</details>

Now that we've set up a `VectorIndex`, we can connect this index with an LLM in a number of ways.
A simple way to do that is with the `SequentialModel`. The first part of the `SequentialModel`
executes a query and provides the results to the LLM in the second part. 

The `RetrievalPrompt` component takes a query with a "free" `Variable` as input. 
This gives users great flexibility with regard to how they fetch the context
for their downstream models.

We're using OpenAI, but you can use any type of LLm with `superduperdb`. We have several 
native integrations (see [here](../ai_integraitons/)) but you can also [bring your own model](../models/bring_your_own_model.md).

```python
from superduperdb.ext.llm.prompter import *
from superduperdb.base.variables import Variable
from superduperdb import Document
from superduperdb.components.model import SequentialModel
from superduperdb.ext.openai import OpenAIChatCompletion

q = db['docu'].like(Document({'txt': Variable('prompt')}), vector_index='my-index', n=5).find().limit(10)

def get_output(c):
    return [r['txt'] for r in c]

prompt_template = RetrievalPrompt('my-prompt', select=q, postprocess=Code.from_object(get_output))

llm = OpenAIChatCompletion(
    'gpt-3.5-turbo',
    client_kwargs={'api_key': '[OPENAI-API-KEY]'},
)
seq = SequentialModel('rag', models=[prompt_template, llm])

db.apply(seq)
```

<details>
<summary>Outputs</summary>
<pre>
    from superduperdb import code
    
    @code
    def get_output(c):
        return [r['txt'] for r in c]
    

</pre>
<pre>
    ([],
     SequentialModel(identifier='rag', uuid='1c211bfe-00df-4cff-b2a3-7722265150ca', signature='**kwargs', datatype=None, output_schema=None, flatten=False, model_update_kwargs=\{\}, predict_kwargs=\{\}, compute_kwargs=\{\}, validation=None, metric_values=\{\}, models=[RetrievalPrompt(identifier='my-prompt', uuid='e7142d22-e2fc-44de-8c12-5a8b985239bb', signature='**kwargs', datatype=None, output_schema=None, flatten=False, model_update_kwargs=\{\}, predict_kwargs=\{\}, compute_kwargs=\{\}, validation=None, metric_values=\{\}, preprocess=None, postprocess=Code(identifier='', uuid='f49a9d39-693b-4504-852c-6a682bca7e0c', code="from superduperdb import code\n\n@code\ndef get_output(c):\n    return [r['txt'] for r in c]\n"), select=docu.like(documents[0], vector_index="my-index", n=5).find().limit(10), prompt_explanation="HERE ARE SOME FACTS SEPARATED BY '---' IN OUR DATA REPOSITORY WHICH WILL HELP YOU ANSWER THE QUESTION.", prompt_introduction='HERE IS THE QUESTION WHICH YOU SHOULD ANSWER BASED ONLY ON THE PREVIOUS FACTS:', join='\n---\n'), OpenAIChatCompletion(identifier='gpt-3.5-turbo', uuid='41b0db16-8269-46fa-a6ac-9e56636f68c0', signature='singleton', datatype=None, output_schema=None, flatten=False, model_update_kwargs=\{\}, predict_kwargs=\{\}, compute_kwargs=\{\}, validation=None, metric_values=\{\}, model='gpt-3.5-turbo', max_batch_size=8, openai_api_key=None, openai_api_base=None, client_kwargs=\{'api_key': '[OPENAI-API-KEY]'\}, batch_size=1, prompt='')]))
</pre>
</details>

Now we can test the `SequentialModel` with a sample question:

```python
seq.predict_one('Tell be about vector-indexes?')
```

<details>
<summary>Outputs</summary>
<pre>
    2024-Jun-02 14:31:27.41| INFO     | Duncans-MBP.fritz.box| superduperdb.base.datalayer:1055 | \{\}

</pre>
<pre>
    Batches:   0%|          | 0/1 [00:00\<?, ?it/s]
</pre>
<pre>
    'VectorIndexes in SuperDuperDB wrap a Listener so that outputs are searchable. They can take a second Listener for multimodal search and apply to Listener instances containing Model instances that output vectors, arrays, or tensors. They can be leveraged in SuperDuperDB queries with the `.like` operator. VectorIndexes are set up by applying them to the datalayer `db`.'
</pre>
</details>
