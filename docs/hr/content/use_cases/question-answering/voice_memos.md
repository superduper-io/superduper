# Voice-Memo Assistant on MongoDB (RAG)

## Cataloguing voice-memos for a self managed personal assistant

Explore the capabilities of SuperDuperDB by effortlessly integrating models across various data modalities, including audio and text. This project aims to develop sophisticated data-based applications with minimal code complexity.

### Objectives:

1. Manage a database of audio recordings.
2. Index the content of these audio recordings.
3. Perform searches and queries on the content of these audio recordings.

### Our approach involves:

* Using a transformers model from Facebook's AI team for audio-to-text transcription.
* Applying an OpenAI vectorization model to index the transcribed text.
* Combining the OpenAI ChatGPT model with relevant recordings to query the audio database.

Real-life use cases encompass personal note-taking, voice diaries, meeting transcriptions, language learning, task reminders, podcast indexing, knowledge base creation, journalism interviews, storytelling archives, and music catalog searches.

In this example, we'll organize and catalog voice memos for a self-managed personal assistant using SuperDuperDB.

## Prerequisites

Before diving into the implementation, ensure that you have the necessary libraries installed by running the following commands:

```bash
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

First, we need to establish a connection to a MongoDB datastore via SuperDuperDB. You can configure the `MongoDB_URI` based on your specific setup. Here are some examples of MongoDB URIs:

* For testing (default connection): `mongomock://test`
* Local MongoDB instance: `mongodb://localhost:27017`
* MongoDB with authentication: `mongodb://superduper:superduper@mongodb:27017/documents`
* MongoDB Atlas: `mongodb+srv://<username>:<password>@<atlas_cluster>/<database>`

```python
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
import os

mongodb_uri = os.getenv("MONGODB_URI","mongomock://test")

# Superdupers your database
db = superduper(mongodb_uri)

# Create a collection for Voice memos
voice_collection = Collection('voice-memos')
```

## Load Dataset

In this example, we use the `LibriSpeech` dataset as our voice recording dataset, containing around 1000 hours of read English speech. Similar functionality can be achieved with any audio source, including audio hosted on the web or in an `s3` bucket. For instance, repositories of audio from conference calls or memos can be indexed in the same way.

```python
from datasets import load_dataset
from superduperdb.ext.numpy import array
from superduperdb import Document

# Load the LibriSpeech ASR demo data from Hugging Face datasets
data = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

# Create an `Encoder` for audio data
enc = array('float64', shape=(None,))

# Add the encoder to the SuperDuperDB instance
db.add(enc)

# Insert audio data into the MongoDB collection 'voice_collection'
db.execute(voice_collection.insert_many([
    # Create a SuperDuperDB Document for each audio sample
    Document({'audio': enc(r['audio']['array'])}) for r in data
]))
```

## Install Pre-Trained Model (LibriSpeech) into Database

Apply a pre-trained `transformers` model to the data:

```python
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from superduperdb.ext.transformers import Pipeline

# Load the pre-trained Speech2Text model and processor from Facebook's library
model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

# Define the sampling rate for the audio data
SAMPLING_RATE = 16000

# Create a SuperDuperDB pipeline for speech-to-text transcription
transcriber = Pipeline(
    identifier='transcription',
    object=model,  # The pre-trained Speech2Text model
    preprocess=processor,  # The processor for handling input audio data
    preprocess_kwargs={'sampling_rate': SAMPLING_RATE, 'return_tensors': 'pt', 'padding': True},  # Preprocessing configurations
    postprocess=lambda x: processor.batch_decode(x, skip_special_tokens=True),  # Postprocessing to convert model output to text
    predict_method='generate',  # Specify the prediction method
    preprocess_type='other',  # Specify the type of preprocessing
)
```

## Run Predictions on All Recordings in the Collection

Apply the `Pipeline` to all audio recordings:

```python
transcriber.predict(
    X='audio',  # Specify the input feature name as 'audio'
    db=db,  # Provide the SuperDuperDB instance
    select=voice_collection.find(),  # Specify the collection of audio data to transcribe
    max_chunk_size=10  # Set the maximum chunk size for processing audio data
)
```

## Ask Questions to Your Voice Assistant

Interact with your voice assistant by asking questions, leveraging the capabilities of MongoDB for vector-search and filtering rules:

```python
from superduperdb import VectorIndex, Listener
from superduperdb.ext.openai import OpenAIEmbedding

# Create a VectorIndex with OpenAI embedding for audio transcriptions
db.add(
    VectorIndex(
        identifier='my-index',  # Set a unique identifier for the VectorIndex
        indexing_listener=Listener(
            model=OpenAIEmbedding(identifier= 'text-embedding-ada-002', model='text-embedding-ada-002'),  # Use OpenAIEmbedding for audio transcriptions
            key='_outputs.audio.transcription',  # Specify the key for indexing the transcriptions in the output
            select=voice_collection.find(),  # Select the collection of audio data to index
        ),
    )
)
```

Let's verify the functionality by searching for the term "royal cavern."

```python
# Define the search parameters
search_term = 'royal cavern'  # Set the search term for audio transcriptions
num_results = 2  # Set the number of desired search results

# Execute a search query using the VectorIndex 'my-index'
# Search for audio transcriptions similar to the specified search term
# and retrieve the specified number of results
search_results = list(
    db.execute(
        voice_collection.like(
            {'_outputs.audio.transcription': search_term},
            n=num_results,
            vector_index='my-index',  # Use the 'my-index' VectorIndex for similarity search
        ).find({}, {'_outputs.audio.transcription': 1})  # Retrieve only the 'transcription' field in the results
    )
)
```

## Enrich with Chat-Completion

Connect the previous steps with gpt-3.5.turbo, a chat-completion model on OpenAI. The goal is to enhance completions by seeding them with the most relevant audio recordings, determined by their textual transcriptions. Retrieve these transcriptions using the previously configured `VectorIndex`.

```python
# Import the OpenAIChatCompletion module from superduperdb.ext.openai
from superduperdb.ext.openai import OpenAIChatCompletion

# Create an instance of OpenAIChatCompletion with the GPT-3.5-turbo model
chat = OpenAIChatCompletion(
    model='gpt-3.5-turbo',
    prompt=(
        'Use the following facts to answer this question\n'
        '{context}\n\n'
        'Here\'s the question:\n'
    ),
)

# Add the OpenAIChatCompletion instance to the database
db.add(chat)

# Display the details of the added model in the database
print(db.show('model'))
```

## Full Voice-Assistant Experience

Evaluate the complete model by asking a question related to a specific fact mentioned in the audio recordings. The model will retrieve the most relevant recordings and utilize them to formulate its answer:

```python
from superduperdb import Document

# Define a question to ask the chat completion model
question = 'Is anything really Greek?'

# Use the db.predict method to get a response from the GPT-3.5-turbo model
response = db.predict(
    model_name='gpt-3.5-turbo',
    
    # Input the question to the chat completion model
    input=question,
    
    # Select relevant context for the model from the SuperDuperDB collection of audio transcriptions
    context_select=voice_collection.like(
        Document({'_outputs.audio.transcription': question}), vector_index='my-index'
    ).find(),
    
    # Specify the key in the context used by the model
    context_key='_outputs.audio.transcription',
)[0].content

# Print the response obtained from the chat completion model
print(response)
```