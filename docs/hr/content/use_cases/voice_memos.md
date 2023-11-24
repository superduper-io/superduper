# Voice Memo Cataloging

## Cataloguing voice-memos for a self managed personal assistant

Discover the magic of SuperDuperDB as we seamlessly integrate models across different data modalities, such as audio and text. Experience the creation of highly sophisticated data-based applications with minimal boilerplate code.

### Objectives:

1. Maintain a database of audio recordings
2. Index the content of these audio recordings
3. Search and interrogate the content of these audio recordings

### Our approach involves:

* Utilizing a transformers model by Facebook's AI team to transcribe audio to text.
* Employing an OpenAI vectorization model to index the transcribed text.
* Harnessing OpenAI ChatGPT model in conjunction with relevant recordings to query the audio database.

## Prerequisites

Before diving into the implementation, ensure that you have the necessary libraries installed by running the following commands:


```python
!pip install superduperdb
!pip install transformers soundfile torchaudio librosa openai
!pip install -U datasets
```

Additionally, ensure that you have set your openai API key as an environment variable. You can uncomment the following code and add your API key:


```python
import os

#os.environ['OPENAI_API_KEY'] = 'sk-XXXX'

if 'OPENAI_API_KEY' not in os.environ:
    raise Exception('Environment variable "OPENAI_API_KEY" not set')
```

## Connect to datastore 

First, we need to establish a connection to a MongoDB datastore via SuperDuperDB. You can configure the `MongoDB_URI` based on your specific setup. 
Here are some examples of MongoDB URIs:

* For testing (default connection): `mongomock://test`
* Local MongoDB instance: `mongodb://localhost:27017`
* MongoDB with authentication: `mongodb://superduper:superduper@mongodb:27017/documents`
* MongoDB Atlas: `mongodb+srv://<username>:<password>@<atlas_cluster>/<database>`


```python
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
import os

mongodb_uri = os.getenv("MONGODB_URI","mongomock://test")
db = superduper(mongodb_uri)

# Create a collection for Voice memos
voice_collection = Collection('voice-memos')
```


## Load Dataset

In this example se use `LibriSpeech` as our voice recording dataset. It is a corpus of approximately 1000 hours of read English speech. The same functionality could be accomplised using any audio, in particular audio hosted on the web, or in an `s3` bucket. For instance, if you have a repository of audio of conference calls, or memos, this may be indexed in the same way. 


```python
from datasets import load_dataset
from superduperdb.ext.numpy import array
from superduperdb import Document

data = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

# Using an `Encoder`, we may add the audio data directly to a MongoDB collection:
enc = array('float64', shape=(None,))

db.add(enc)

db.execute(voice_collection.insert_many([
    Document({'audio': enc(r['audio']['array'])}) for r in data
]))
```

## Install Pre-Trained Model (LibreSpeech) into Database

Apply a pretrained `transformers` model to the data: 


```python
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from superduperdb.ext.transformers import Pipeline

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

SAMPLING_RATE = 16000

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

# Run Predictions on All Recordings in the Collection
Apply the `Pipeline` to all audio recordings:


```python
transcriber.predict(X='audio', db=db, select=voice_collection.find(), max_chunk_size=10)
```

## Ask Questions to Your Voice Assistant

Ask questions to your voice assistant, targeting specific queries and utilizing the power of MongoDB for vector-search and filtering rules:


```python
from superduperdb import VectorIndex, Listener
from superduperdb.ext.openai import OpenAIEmbedding

db.add(
    VectorIndex(
        identifier='my-index',
        indexing_listener=Listener(
            model=OpenAIEmbedding(model='text-embedding-ada-002'),
            key='_outputs.audio.transcription',
            select=voice_collection.find(),
        ),
    )
)
```

Let's confirm this has worked, by searching for the `royal cavern`.


```python
# Define the search parameters
search_term = 'royal cavern'
num_results = 2

list(db.execute(
    voice_collection.like(
        {'_outputs.audio.transcription': search_term},
        n=num_results,
        vector_index='my-index',
    ).find({}, {'_outputs.audio.transcription': 1})
))
```

## Enrich it with Chat-Completion 

Connect the previous steps with the gpt-3.5.turbo, a chat-completion model on OpenAI. The plan is to seed the completions with the most relevant audio recordings, as judged by their textual transcriptions. These transcriptions are retrieved using the previously configured `VectorIndex`. 


```python
from superduperdb.ext.openai import OpenAIChatCompletion

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

## Full Voice-Assistant Experience

Test the full model by asking a question about a specific fact mentioned in the audio recordings. The model will retrieve the most relevant recordings and use them to formulate its answer:



```python
from superduperdb import Document

q = 'Is anything really Greek?'

print(db.predict(
    model_name='gpt-3.5-turbo',
    input=q,
    context_select=voice_collection.like(
        Document({'_outputs.audio.transcription': q}), vector_index='my-index'
    ).find(),
    context_key='_outputs.audio.transcription',
)[0].content)
```
