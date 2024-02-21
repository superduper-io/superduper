
# Q&A Assistant Using OpenAI on MongoDB (RAG)

This notebook is designed to demonstrate how to implement a document Question-and-Answer (Q&A) task using SuperDuperDB in conjunction with OpenAI and MongoDB. It provides a step-by-step guide and explanation of each component involved in the process.

Implementing a document Question-and-Answer (Q&A) system using SuperDuperDB, OpenAI, and MongoDB can find applications in various real-life scenarios:

1. **Customer Support Chatbots:** Enable a chatbot to answer customer queries by extracting information from documents, manuals, or knowledge bases stored in MongoDB or any other SuperDuperDB supported database using Q&A.

2. **Legal Document Analysis:** Facilitate legal professionals in quickly extracting relevant information from legal documents, statutes, and case laws, improving efficiency in legal research.

3. **Medical Data Retrieval:** Assist healthcare professionals in obtaining specific information from medical documents, research papers, and patient records for quick reference during diagnosis and treatment.

4. **Educational Content Assistance:** Enhance educational platforms by enabling students to ask questions related to course materials stored in a MongoDB database, providing instant and accurate responses.

5. **Technical Documentation Search:** Support software developers and IT professionals in quickly finding solutions to technical problems by querying documentation and code snippets stored in MongoDB or any other database supported by SuperDuperDB. We did that!

6. **HR Document Queries:** Simplify HR processes by allowing employees to ask questions about company policies, benefits, and procedures, with answers extracted from HR documents stored in MongoDB or any other database supported by SuperDuperDB.

7. **Research Paper Summarization:** Enable researchers to pose questions about specific topics, automatically extracting relevant information from a MongoDB repository of research papers to generate concise summaries.

8. **News Article Information Retrieval:** Empower users to inquire about specific details or background information from a database of news articles stored in MongoDB or any other database supported by SuperDuperDB, enhancing their understanding of current events.

9. **Product Information Queries:** Improve e-commerce platforms by allowing users to ask questions about product specifications, reviews, and usage instructions stored in a MongoDB database.

By implementing a document Q&A system with SuperDuperDB, OpenAI, and MongoDB, these use cases demonstrate the versatility and practicality of such a solution across different industries and domains.

All is possible without zero friction with SuperDuperDB. Now back into the notebook.

## Prerequisites

Before starting the implementation, make sure you have the required libraries installed by running the following commands:

```bash
!pip install superduperdb
!pip install ipython openai==1.1.2
```

Additionally, ensure that you have set your OpenAI API key as an environment variable. You can uncomment the following code and add your API key:

```python
import os

# Add your OPEN_AI_API_KEY
# os.environ['OPENAI_API_KEY'] = 'sk-...'

if 'OPENAI_API_KEY' not in os.environ:
    raise Exception('Environment variable "OPENAI_API_KEY" not set')
```

## Connect to Datastore

First, we need to establish a connection to a MongoDB datastore via SuperDuperDB. You can configure the `MongoDB_URI` based on your specific setup.

Here are some examples of MongoDB URIs:

- For testing (default connection): `mongomock://test`
- Local MongoDB instance: `mongodb://localhost:27017`
- MongoDB with authentication: `mongodb://superduper:superduper@mongodb:27017/documents`
- MongoDB Atlas: `mongodb+srv://<username>:<password>@<atlas_cluster>/<database>`

```python
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
import os

mongodb_uri = os.getenv("MONGODB_URI", "mongomock://test")

# SuperDuperDB, now handles your MongoDB database
# It just super dupers your database 
db = superduper(mongodb_uri)

collection = Collection('questiondocs')
```

```python
db.metadata
```

## Load Dataset

In this example, we use the internal textual data from the `superduperdb` project's API documentation. The objective is to create a chatbot that can offer information about the project. You can either load the data from your local project or use the provided data.

If you have the SuperDuperDB project locally and want to load the latest version of the API, uncomment the following cell:

```python
# import glob

# ROOT = '../docs/hr/content/docs/'
# STRIDE = 3       # stride in numbers of lines
# WINDOW = 25       # length of window in numbers of lines

# files = sorted(glob.glob(f'{ROOT}/*.md') + glob.glob(f'{ROOT}/*.mdx'))

# content = sum([open(file).read().split('\n') for file in files], [])
# chunks = ['\n'.join(content[i: i + WINDOW]) for i in range(0, len(content), STRIDE)]
```

Otherwise, you can load the data from an external source. The text chunks include code snippets and explanations, which will be utilized to construct the document Q&A chatbot.

```python
from IPython.display import *

# Assuming 'chunks' is a list or iterable containing markdown content
Markdown(chunks[20])
```

```python
# Use !curl to download the 'superduperdb_docs.json' file
!curl -O https://superduperdb-public.s3.eu-west-1.amazonaws.com/superduperdb_docs.json

import json
from IPython.display import Markdown

# Open the downloaded JSON file and load its contents into the 'chunks' variable
with open('superduperdb_docs.json') as f:
    chunks = json.load(f)
```

The chunks of text contain both code snippets and explanations, making them valuable for constructing a document Q&A chatbot. The combination of code and explanations enables the chatbot to provide comprehensive and context-aware responses to user queries.

As usual, we insert the data. The `Document` wrapper allows `superduperdb` to handle records with special data types such as images, video, and custom data-types.

```python
from superduperdb import Document

# Insert multiple documents into the collection
db.execute(collection.insert_many([Document({'txt': chunk}) for chunk in chunks]))
```

## Create a Vector-Search Index

To enable question-answering over your documents, set up a standard `superduperdb` vector-search index using `openai` (other options include `torch`, `sentence_transformers`, `transformers`, etc.).

A `Model` is a wrapper around a self-built or ecosystem model, such as `torch`, `transformers`, `openai`.

```python
from superduperdb.ext.openai import OpenAIEmbedding

# Create an instance of the OpenAIEmbedding model with the specified identifier ('text-embedding-ada-002')
model = OpenAIEmbedding(identifier= 'text-embedding-ada-002', model='text-embedding-ada-002')
```

```python
model.predict('This is a test', one=True)
```

A `Listener` essentially deploys a `Model` to "listen" to incoming data, computes outputs, and then saves the results in the database via

 `db`.

```python
# Import the Listener class from the superduperdb module
from superduperdb import Listener

# Create a Listener instance with the specified model, key, and selection criteria
listener = Listener(
    model=model,          # The model to be used for listening
    key='txt',            # The key field in the documents to be processed by the model
    select=collection.find()  # The selection criteria for the documents
)
```

A `VectorIndex` wraps a `Listener`, allowing its outputs to be searchable.

```python
# Import the VectorIndex class from the superduperdb module
from superduperdb import VectorIndex

# Add a VectorIndex to the SuperDuperDB database with the specified identifier and indexing listener
db.add(
    VectorIndex(
        identifier='my-index',        # Unique identifier for the VectorIndex
        indexing_listener=listener    # Listener to be used for indexing documents
    )
)
```

```python
# Execute a find_one operation on the SuperDuperDB collection
db.execute(collection.find_one())
```

```python
from superduperdb.backends.mongodb import Collection
from superduperdb import Document as D
from IPython.display import *

# Define the query for the search
query = 'Code snippet how to create a `VectorIndex` with a torchvision model'

# Execute a search using SuperDuperDB to find documents containing the specified query
result = db.execute(
    collection
        .like(D({'txt': query}), vector_index='my-index', n=5)
        .find()
)

# Display a horizontal rule to separate results
display(Markdown('---'))

# Display each document's 'txt' field and separate them with a horizontal rule
for r in result:
    display(Markdown(r['txt']))
    display(Markdown('---'))
```

## Create a Chat-Completion Component

In this step, a chat-completion component is created and added to the system. This component is essential for the Q&A functionality:

```python
# Import the OpenAIChatCompletion class from the superduperdb.ext.openai module
from superduperdb.ext.openai import OpenAIChatCompletion

# Define the prompt for the OpenAIChatCompletion model
prompt = (
    'Use the following description and code snippets about SuperDuperDB to answer this question about SuperDuperDB\n'
    'Do not use any other information you might have learned about other python packages\n'
    'Only base your answer on the code snippets retrieved\n'
    '{context}\n\n'
    'Here\'s the question:\n'
)

# Create an instance of OpenAIChatCompletion with the specified model and prompt
chat = OpenAIChatCompletion(model='gpt-3.5-turbo', prompt=prompt)

# Add the OpenAIChatCompletion instance
db.add(chat)

# Print information about the models in the SuperDuperDB database
print(db.show('model'))
```

## Ask Questions to Your Docs

Finally, you can ask questions about the documents. You can target specific queries and use the power of MongoDB for vector-search and filtering rules. Here's an example of asking a question:

```python
from superduperdb import Document
from IPython.display import Markdown

# Define the search parameters
search_term = 'Can you give me a code-snippet to set up a `VectorIndex`?'
num_results = 5

# Use the SuperDuperDB model to generate a response based on the search term and context
output, context = db.predict(
    model_name='gpt-3.5-turbo',
    input=search_term,
    context_select=(
        collection
            .like(Document({'txt': search_term}), vector_index='my-index', n=num_results)
            .find()
    ),
    context_key='txt',
)

# Display the generated response using Markdown
Markdown(output.content)
```

## Reset the Demo

```python
# Remove a VectorIndex with the identifier 'my-index'
db.remove('vector_index', 'my-index', force=True)

# Remove a Listener associated with the 'text-embedding-ada-002/txt' key
db.remove('listener', 'text-embedding-ada-002/txt', force=True)

# Remove a model with the identifier 'text-embedding-ada-002'
db.remove('model', 'text-embedding-ada-002', force=True)
```

## Now you can build an API as well just like we did

### FastAPI Question the Docs Apps Tutorial

This tutorial will guide you through setting up a basic FastAPI application for handling questions with documentation. The tutorial covers both local development and deployment to the Fly.io platform.
[https://github.com/SuperDuperDB/chat-with-your-docs-backend](https://github.com/SuperDuperDB/chat-with-your-docs-backend)