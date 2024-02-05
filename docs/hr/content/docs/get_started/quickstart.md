---
sidebar_position: 1
---

# Quickstart

Welcome and get ready to add AI to your Database in minutes. 

## Setup

### Jupyter Notebook

We recommend using Jupyter Notebooks or Google Colabs for a quicksatrt due to their interactive nature, which is helpful for understanding and troubleshooting. While not mandatory, it enhances the learning experience. Installation instructions for Jupyter Notebooks are provided [here](https://jupyter.org/install).

### Installation

There are two ways to get started:

#### Pip
```
pip install superduperdb
```
Make sure to have `python3.10` or `python3.11`

#### Docker

```
docker run -p 8888:8888 superduperdb/superduperdb:latest
```

For more details on installation and control, see our [Installation guide](./installation.md).

## Adding AI to your classical database

SuperDuper enables integrating your database of choice with AI models, APIs, and vector search engines, providing streaming inference and scalable training/fine-tuning. It integrates with popular classical databases like MongoDB, Postgres, Snowflakes, MySQL etc
Check [here](../data_integrations/) for our available range of database integrations. 

It also integrates with popular embedding models like OpenAI , Cohere etc. Check [here](../ai_integrations/) for our available range of model integrations.

In this quickstart, we will walk through extending the functionality of MongoDB by adding an OPENAI embedding model into the database and then adding a vector index easily in order to do a quick simple semantic search

For this demo, 
- we would be using MongoDB Test DB url:  `mongomock://test`.
- We will also be using a json toy dataset below with titles and descriptions relating to politics, sport and health
```
data = [
  {
    "title": "Election Results",
    "description": "Detailed analysis of recent election results and their implications."
  },
  {
    "title": "Foreign Relations",
    "description": "Discussion on current diplomatic relations with neighboring countries and global partners."
  },
  {
    "title": "Policy Changes",
    "description": "Overview of proposed policy changes and their potential impact on the population."
  },
  {
    "title": "Championship Game",
    "description": "Recap of the thrilling championship game, including key plays and player performances."
  },
  {
    "title": "Athlete Spotlight",
    "description": "Profile of a prominent athlete, highlighting their achievements and career milestones."
  },
  {
    "title": "Upcoming Tournaments",
    "description": "Preview of upcoming sports tournaments, schedules, and participating teams."
  },
  {
    "title": "COVID-19 Vaccination Drive",
    "description": "Updates on the progress of the COVID-19 vaccination campaign and vaccination centers."
  },
  {
    "title": "Mental Health Awareness",
    "description": "Importance of mental health awareness and tips for maintaining emotional well-being."
  },
  {
    "title": "Healthy Eating Habits",
    "description": "Nutritional advice and guidelines for maintaining a balanced and healthy diet."
  }
]

```



### Let's Start

First we'll need to install the  OpenAI  package.

```
pip install openai
```

Accessing the OPENAI API requires an API key, which you can get by creating an account and heading [here](https://platform.openai.com/account/api-keys).

Once we have a key, we'll want to set it as an environment variable by running:

```
import os

os.environ['OPENAI_API_KEY'] = 'sk-...'
```

we intialize the model
```
from superduperdb.ext.openai.model import OpenAIEmbedding

openai_model = OpenAIEmbedding(model='text-embedding-ada-002')

```
 

Next, we convert our DB into a SuperDuper object to expand the functionality 

```
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
import os

mongodb_uri = os.getenv("MONGODB_URI","mongomock://test")

db = superduper(mongodb_uri)

doc_collection = Collection('documents')

```

_Feel free to check the urls to your mongodb atlas url with existing datasets in it_

Next , we insert the data
```
from superduperdb import Document


db.execute(doc_collection.insert_many([Document(r) for r in data]))

```
_Note that if you already have data in your database, you can skip the **"insert the data"** part_


Next, we add the vector search functionality and add our model to the DB to convert all text in the database into vectors and also convery incoming queries into vectors

```
from superduperdb import Listener, VectorIndex

db.add(
    VectorIndex(
       
        identifier=f'mongodb-data-{openai_model.identifier}',
        indexing_listener=Listener(
            select=doc_collection.find(),  
            key='description', 
            model=openai_model,  
            
        ),
    )
)

```
Now, we are ready to use vector search on the database. For this demo, we would limit the search results to 2. 

_Feel free to increase the limit_

```
user_query = 'sport articles'
limit_search_results = 2


result = db.execute(
    doc_collection
        .like(Document({'description': user_query}), vector_index=f'mongodb-data-{openai_model.identifier}', n=limit_search_results)
        .find({}, {'title': 1, 'description': 1, 'score': 1})
)

```

To view the result
```
for r in result:
  print(r.unpack())

```

We've now successfully extended your database functionality to be able to convert incoming query to vectors and then use your the query vectors to search for similar info in your existing database . In other words, adding AI to your database



## Diving Deeper
- Explore more of the documenation  and check out other [`use-cases`](/docs/use-cases)
- Checkout how you can use SuperDuper to [`set-up`](/docs/category/setup) more configurations
- Experiment with other available database integrations and model/AI intergration
- Dig deeper into other of our[`fundamental frameworks`](../fundamentals/glossary) apart from vector search. Also check out the [`API references`](https://docs.superduperdb.com/apidocs/source/superduperdb.html) of the source codes.


## Engage with the Community

SuperDuperDB is a community effort and licensed under Apache 2.0. We encourage enthusiastic developers to contribute to the project

- Visit our [`community apps example`](https://github.com/superDuperDB/superduper-community-apps) repository to explore more examples of how SuperDuperDB can enhance your experience. Learn from real-world use cases and implementations.

- Want to add some of your favourite integrations to SuperDuper? You are welcome to join the conversation on our [`discussions forum`](https://github.com/SuperDuperDB/superduperdb/discussions) and follow our open-source roadmap [`here`](https://github.com/orgs/SuperDuperDB/projects/1/views/10).

- If you encounter challenges, join our [`Slack Channels`](https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA) for assistance. Report bugs and share feature requests [`by raising an issue`]((https://github.com/SuperDuperDB/superduperdb/issues).). 
