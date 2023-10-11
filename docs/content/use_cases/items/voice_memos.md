# Cataloguing voice-memos for a self managed personal assistant

In this example we show-case SuperDuperDB's ability to combine models across data modalities, 
in this case audio and text, to devise highly sophisticated data based apps, with very little 
boilerplate code.

The aim, is to:

- Maintain a database of audio recordings
- Index the content of these audio recordings
- Search and interrogate the content of these audio recordings

We accomplish this by:

1. Use a `transformers` model by Facebook's AI team to transcribe the audio to text
2. Use an OpenAI vectorization model to index the transcribed text
3. Use OpenAI's ChatGPT model in combination with relevant recordings to interrogate the contents
  of the audio database


```python
!pip install superduperdb==0.0.12
!pip install torchaudio==2.1.0
!pip install datasets==2.10.1   # 2.14 seems to be broken so rolling back version
```

This functionality could be accomplised using any audio, in particular audio 
hosted on the web, or in an `s3` bucket. For instance, if you have a repository
of audio of conference calls, or memos, this may be indexed in the same way.

To make matters simpler, we use a dataset of audio recordings from the `datasets` library, to demonstrate the 
functionality:


```python
from datasets import load_dataset

SAMPLING_RATE = 16000

data = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
```

As usual we wrap our MongoDB connector, to connect to the `Datalayer`:


```python
import os
from superduperdb import superduper

# Uncomment one of the following lines to use a bespoke MongoDB deployment
# For testing the default connection is to mongomock

mongodb_uri = os.getenv("MONGODB_URI","mongomock://test")
# mongodb_uri = "mongodb://localhost:27017"
# mongodb_uri = "mongodb://superduper:superduper@mongodb:27017/documents"
# mongodb_uri = "mongodb://<user>:<pass>@<mongo_cluster>/<database>"
# mongodb_uri = "mongodb+srv://<username>:<password>@<atlas_cluster>/<database>"

# Super-Duper your Database!
from superduperdb import superduper
db = superduper(mongodb_uri)
```

Using an `Encoder`, we may add the audio data directly to a MongoDB collection:


```python
from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.numpy.array import array
from superduperdb.container.document import Document as D

collection = Collection('voice-memos')
enc = array('float32', shape=(None,))

db.execute(collection.insert_many([
    D({'audio': enc(r['audio']['array'])}) for r in data
], encoders=(enc,)))
```

Now that we've done that, let's apply a pretrained `transformers` model to this data:


```python
import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
```

We wrap this model using the SuperDuperDB wrapper for `transformers`:


```python
from superduperdb.ext.transformers.model import Pipeline

transcriber = Pipeline(
    identifier='transcription',
    object=model,
    preprocess=processor,
    preprocess_kwargs={'sampling_rate': SAMPLING_RATE, 'return_tensors': 'pt', 'padding': True},
    postprocess=lambda x: processor.batch_decode(x, skip_special_tokens=True),
    predict_method='generate',
    preprocess_type='other',
)
```

Let's verify this `Pipeline` works on a sample datapoint


```python
import IPython

IPython.display.Audio(data[0]['audio']['array'], rate=SAMPLING_RATE)
```


```python
transcriber.predict(data[0]['audio']['array'], one=True)
```

Now let's apply the `Pipeline` to all of the audio recordings:


```python
transcriber.predict(X='audio', db=db, select=collection.find(), max_chunk_size=10)
```

We may now verify that all of the recordings have been transcribed in the database


```python
list(db.execute(
    Collection('voice-memos').find({}, {'_outputs.audio.transcription': 1})
))
```

As in previous examples, we can use OpenAI's text-embedding models to vectorize and search the 
textual transcriptions directly in MongoDB:


```python
import os
os.environ['OPENAI_API_KEY'] = '<YOUR-API-KEY>'
```


```python
from superduperdb.container.vector_index import VectorIndex
from superduperdb.container.listener import Listener
from superduperdb.ext.openai.model import OpenAIEmbedding
from superduperdb.db.mongodb.query import Collection

db.add(
    VectorIndex(
        identifier='my-index',
        indexing_listener=Listener(
            model=OpenAIEmbedding(model='text-embedding-ada-002'),
            key='_outputs.audio.transcription',
            select=Collection(name='voice-memos').find(),
        ),
    )
)
```

Let's confirm this has worked, by searching for the "royal cavern"


```python
list(db.execute(
    Collection('voice-memos').like(
        {'_outputs.audio.transcription': 'royal cavern'},
        n=2,
        vector_index='my-index',
    ).find({}, {'_outputs.audio.transcription': 1})
))
```

Now we can connect the previous steps with the `gpt-3.5.turbo`, which is a chat-completion 
model on OpenAI. The plan is to seed the completions with the most relevant audio recordings, 
as judged by their textual transcriptions. These transcriptions are retrieved using 
the previously configured `VectorIndex`:


```python
from superduperdb.ext.openai.model import OpenAIChatCompletion

chat = OpenAIChatCompletion(
    model='gpt-3.5-turbo',
    prompt=(
        'Use the following facts to answer this question\n'
        '{context}\n\n'
        'Here\'s the question:\n'
    ),
)

db.add(chat)

print(db.show('model'))
```

Let's test the full model! We can ask a question which asks about a specific fact 
mentioned somewhere in the audio recordings. The model will retrieve the most relevant
recordings, and use these in formulating its answer:


```python
from superduperdb.container.document import Document

q = 'Is anything really Greek?'

print(db.predict(
    model='gpt-3.5-turbo',
    input=q,
    context_select=Collection('voice-memos').like(
        Document({'_outputs.audio.transcription': q}), vector_index='my-index'
    ).find(),
    context_key='_outputs.audio.transcription',
)[0].content)
```
