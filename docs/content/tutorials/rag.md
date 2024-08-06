
# Basic RAG tutorial

:::info
In this tutorial we show you how to do retrieval augmented generation (RAG) with `superduper`.
Note that this is just an example of the flexibility and power which `superduper` gives 
to developers. `superduper` is about much more than RAG and LLMs. 
:::

As in the vector-search tutorial we'll use `superduper` documentation for the tutorial.
We'll add this to a testing database by downloading the data snapshot:

```python
!curl -O https://superduper-public-demo.s3.amazonaws.com/text.json
```

<details>
<summary>Outputs</summary>
<pre>
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  720k  100  720k    0     0   679k      0  0:00:01  0:00:01 --:--:--  681k

</pre>
</details>

```python
import json

from superduper import superduper, Document

db = superduper('mongomock://test')

with open('text.json') as f:
    data = json.load(f)

_ = db['docu'].insert_many([{'txt': r} for r in data]).execute()
```

<details>
<summary>Outputs</summary>

</details>

Let's verify the data in the `db` by querying one datapoint:

```python
db['docu'].find_one().execute()
```

<details>
<summary>Outputs</summary>

</details>

The first step in a RAG application is to create a `VectorIndex`. The results of searching 
with this index will be used as input to the LLM for answering questions.

Read about `VectorIndex` [here](../apply_api/vector_index.md) and follow along the tutorial on 
vector-search [here](./vector_search.md).

```python
import requests 

from superduper import Application, Document, VectorIndex, Listener, vector
from superduper.ext.sentence_transformers.model import SentenceTransformer
from superduper.base.code import Code

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

</details>

Now that we've set up a `VectorIndex`, we can connect this index with an LLM in a number of ways.
A simple way to do that is with the `SequentialModel`. The first part of the `SequentialModel`
executes a query and provides the results to the LLM in the second part. 

The `RetrievalPrompt` component takes a query with a "free" variable as input, signified with `<var:???>`. 
This gives users great flexibility with regard to how they fetch the context
for their downstream models.

We're using OpenAI, but you can use any type of LLm with `superduper`. We have several 
native integrations (see [here](../ai_integraitons/)) but you can also [bring your own model](../models/bring_your_own_models.md).

```python
from superduper.ext.llm.prompter import *
from superduper import Document
from superduper.components.model import SequentialModel
from superduper.ext.openai import OpenAIChatCompletion

q = db['docu'].like(Document({'txt': '<var:prompt>'}), vector_index='my-index', n=5).find().limit(10)

def get_output(c):
    return [r['txt'] for r in c]

prompt_template = RetrievalPrompt('my-prompt', select=q, postprocess=Code.from_object(get_output))

llm = OpenAIChatCompletion('gpt-3.5-turbo')
seq = SequentialModel('rag', models=[prompt_template, llm])

db.apply(seq)
```

<details>
<summary>Outputs</summary>

</details>

Now we can test the `SequentialModel` with a sample question:

```python
seq.predict('Tell be about vector-indexes')
```

<details>
<summary>Outputs</summary>

</details>

:::tip
Did you know you can use any tools from the Python ecosystem with `superduper`.
That includes `langchain` and `llamaindex` which can be very useful for RAG applications.
:::

```python
from superduper import Application

app = Application('rag-app', components=[vector_index, seq, plugin_1, plugin_2])
```

<details>
<summary>Outputs</summary>

</details>

```python
app.encode()
```

<details>
<summary>Outputs</summary>

</details>

```python
app.export('rag-app')
```

<details>
<summary>Outputs</summary>

</details>

```python
!cat rag-app/requirements.txt
```

<details>
<summary>Outputs</summary>

</details>

```python
from superduper import *

app = Component.read('rag-app')
```

<details>
<summary>Outputs</summary>
<pre>
    /Users/dodo/.pyenv/versions/3.11.7/envs/superduper-3.11/lib/python3.11/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(

</pre>
</details>

```python
app.info()
```

<details>
<summary>Outputs</summary>
<pre>
    2024-Jun-17 09:42:33.43| INFO     | Duncans-MBP.fritz.box| superduper.base.document:362  | Building leaf \<class 'superduper.components.vector_index.VectorIndex'\> with identifier: my-index
    2024-Jun-17 09:42:33.43| INFO     | Duncans-MBP.fritz.box| superduper.base.document:362  | Building leaf \<class 'superduper.components.listener.Listener'\> with identifier: my-listener
    2024-Jun-17 09:42:33.43| INFO     | Duncans-MBP.fritz.box| superduper.base.document:362  | Building leaf \<class 'superduper.ext.sentence_transformers.model.SentenceTransformer'\> with identifier: my-embedding
    2024-Jun-17 09:42:33.44| INFO     | Duncans-MBP.fritz.box| superduper.base.document:362  | Building leaf \<class 'superduper.components.datatype.DataType'\> with identifier: my-vec
    2024-Jun-17 09:42:33.44| INFO     | Duncans-MBP.fritz.box| superduper.base.document:362  | Building leaf \<class 'superduper.base.code.Code'\> with identifier: postprocess
    2024-Jun-17 09:42:33.44| INFO     | Duncans-MBP.fritz.box| superduper.base.document:362  | Building leaf \<class 'superduper.backends.mongodb.query.MongoQuery'\> with identifier: docu-find
    2024-Jun-17 09:42:33.44| INFO     | Duncans-MBP.fritz.box| superduper.base.document:362  | Building leaf \<class 'superduper.components.model.SequentialModel'\> with identifier: rag
    2024-Jun-17 09:42:33.44| INFO     | Duncans-MBP.fritz.box| superduper.base.document:362  | Building leaf \<class 'superduper.ext.llm.prompter.RetrievalPrompt'\> with identifier: my-prompt
    2024-Jun-17 09:42:33.44| INFO     | Duncans-MBP.fritz.box| superduper.base.document:362  | Building leaf \<class 'superduper.base.code.Code'\> with identifier: get_output
    2024-Jun-17 09:42:33.44| INFO     | Duncans-MBP.fritz.box| superduper.base.document:362  | Building leaf \<class 'superduper.backends.mongodb.query.MongoQuery'\> with identifier: docu-like-txt-var-prompt-vector-index-my-index-n-5-find-limit-10
    2024-Jun-17 09:42:33.44| INFO     | Duncans-MBP.fritz.box| superduper.base.document:362  | Building leaf \<class 'superduper.ext.openai.model.OpenAIChatCompletion'\> with identifier: gpt-3.5-turbo

</pre>
<pre>
    [1;32m╭─[0m[1;32m───────────────────────────────────────────────────[0m[1;32m rag-app [0m[1;32m───────────────────────────────────────────────────[0m[1;32m─╮[0m
    [1;32m│[0m [35midentifier[0m: [34mrag-app[0m                                                                                             [1;32m│[0m
    [1;32m│[0m [35muuid[0m: [34m9115f5ec-5575-4a11-8678-664f3904bab7[0m                                                                      [1;32m│[0m
    [1;32m│[0m [35mcomponents[0m: [34m[VectorIndex(identifier='my-index', uuid='650db68c-8786-4204-bc2d-6cc4f1d2511c', [0m                   [1;32m│[0m
    [1;32m│[0m [34mindexing_listener=Listener(identifier='my-listener', uuid='02f5b3d4-7a0a-48d8-990c-bdae29424038', key='txt', [0m   [1;32m│[0m
    [1;32m│[0m [34mmodel=SentenceTransformer(preferred_devices=('cuda', 'mps', 'cpu'), device='cpu', identifier='my-embedding', [0m   [1;32m│[0m
    [1;32m│[0m [34muuid='b1351454-3714-4c57-bacf-2f2a667d5fdc', signature='*args,**kwargs', datatype=DataType(identifier='my-vec',[0m [1;32m│[0m
    [1;32m│[0m [34muuid='ecfbe6d5-5c1f-4b80-b224-aaf0a1f3ee1d', encoder=None, decoder=None, info=None, shape=(384,), [0m              [1;32m│[0m
    [1;32m│[0m [34mdirectory=None, encodable='native', bytes_encoding=\<BytesEncoding.BYTES: 'Bytes'\>, intermediate_type='bytes', [0m  [1;32m│[0m
    [1;32m│[0m [34mmedia_type=None), output_schema=None, flatten=False, model_update_kwargs=\{\}, [0m                                   [1;32m│[0m
    [1;32m│[0m [34mpredict_kwargs=\{'show_progress_bar': True\}, compute_kwargs=\{\}, validation=None, metric_values=\{\}, [0m              [1;32m│[0m
    [1;32m│[0m [34mnum_workers=0, object=SentenceTransformer([0m                                                                      [1;32m│[0m
    [1;32m│[0m [34m  (0): Transformer(\{'max_seq_length': 256, 'do_lower_case': False\}) with Transformer model: BertModel [0m          [1;32m│[0m
    [1;32m│[0m [34m  (1): Pooling(\{'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': [0m  [1;32m│[0m
    [1;32m│[0m [34mTrue, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, [0m                            [1;32m│[0m
    [1;32m│[0m [34m'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True\})[0m            [1;32m│[0m
    [1;32m│[0m [34m  (2): Normalize()[0m                                                                                              [1;32m│[0m
    [1;32m│[0m [34m), model='all-MiniLM-L6-v2', preprocess=None, postprocess=Code(identifier='postprocess', [0m                       [1;32m│[0m
    [1;32m│[0m [34muuid='fadfa78c-4c6b-4914-885a-e1372da93078', code='from superduper import code\n\n@code\ndef [0m                 [1;32m│[0m
    [1;32m│[0m [34mpostprocess(x):\n    return x.tolist()\n')), select=docu.find(), active=True, predict_kwargs=\{'max_chunk_size':[0m [1;32m│[0m
    [1;32m│[0m [34m50\}), compatible_listener=None, measure=\<VectorIndexMeasureType.cosine: 'cosine'\>, metric_values=\{\}), [0m          [1;32m│[0m
    [1;32m│[0m [34mSequentialModel(identifier='rag', uuid='fa46eb15-112c-496f-965f-c935494825c5', signature='**kwargs', [0m           [1;32m│[0m
    [1;32m│[0m [34mdatatype=None, output_schema=None, flatten=False, model_update_kwargs=\{\}, predict_kwargs=\{\}, compute_kwargs=\{\},[0m [1;32m│[0m
    [1;32m│[0m [34mvalidation=None, metric_values=\{\}, num_workers=0, models=[RetrievalPrompt(identifier='my-prompt', [0m              [1;32m│[0m
    [1;32m│[0m [34muuid='ded3b9b8-828d-41a4-bc37-02217fe0bc08', signature='**kwargs', datatype=None, output_schema=None, [0m          [1;32m│[0m
    [1;32m│[0m [34mflatten=False, model_update_kwargs=\{\}, predict_kwargs=\{\}, compute_kwargs=\{\}, validation=None, metric_values=\{\},[0m [1;32m│[0m
    [1;32m│[0m [34mnum_workers=0, preprocess=None, postprocess=Code(identifier='get_output', [0m                                      [1;32m│[0m
    [1;32m│[0m [34muuid='c1d6fb70-b6c7-42b4-8872-8bfd243ddf07', code="from superduper import code\n\n@code\ndef get_output(c):\n[0m [1;32m│[0m
    [1;32m│[0m [34mreturn [r['txt'] for r in c]\n"), select=docu.like(\{'txt': '\<var:prompt\>'\}, vector_index="my-index", [0m           [1;32m│[0m
    [1;32m│[0m [34mn=5).find().limit(10), prompt_explanation="HERE ARE SOME FACTS SEPARATED BY '---' IN OUR DATA REPOSITORY WHICH [0m [1;32m│[0m
    [1;32m│[0m [34mWILL HELP YOU ANSWER THE QUESTION.", prompt_introduction='HERE IS THE QUESTION WHICH YOU SHOULD ANSWER BASED [0m   [1;32m│[0m
    [1;32m│[0m [34mONLY ON THE PREVIOUS FACTS:', join='\n---\n'), OpenAIChatCompletion(identifier='gpt-3.5-turbo', [0m                [1;32m│[0m
    [1;32m│[0m [34muuid='bc04fcdf-3217-4cb7-9517-38fc632fc8f7', signature='singleton', datatype=None, output_schema=None, [0m         [1;32m│[0m
    [1;32m│[0m [34mflatten=False, model_update_kwargs=\{\}, predict_kwargs=\{\}, compute_kwargs=\{\}, validation=None, metric_values=\{\},[0m [1;32m│[0m
    [1;32m│[0m [34mnum_workers=0, model='gpt-3.5-turbo', max_batch_size=8, openai_api_key=None, openai_api_base=None, [0m             [1;32m│[0m
    [1;32m│[0m [34mclient_kwargs=\{\}, batch_size=1, prompt='')])][0m                                                                   [1;32m│[0m
    [1;32m╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯[0m
    [34m╭─[0m[34m─────────────────────────────────────────────[0m[34m Component Metadata [0m[34m──────────────────────────────────────────────[0m[34m─╮[0m
    [34m│[0m [33mVariables[0m                                                                                                       [34m│[0m
    [34m│[0m [35mprompt[0m                                                                                                          [34m│[0m
    [34m│[0m                                                                                                                 [34m│[0m
    [34m│[0m                                                                                                                 [34m│[0m
    [34m│[0m [33mLeaves[0m                                                                                                          [34m│[0m
    [34m╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯[0m

</pre>
</details>
