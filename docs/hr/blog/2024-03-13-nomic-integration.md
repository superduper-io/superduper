---
slug: nomic-integration
title: 'Integrating Nomic API with MongoDB using SuperDuperDB'
authors: [anita]
tags: [Nomic, MongoDB, API]
---

## Integrating Nomic API with MongoDB using SuperDuperDB

![Photo](https://cdn-images-1.medium.com/max/4096/0*2OFjqtMwgV0LEwMm)

One of the major components of building an RAG system is being able to perform a **vector search** or a **semantic search**. This potentially includes having an **embedding model** and **a database of choice**.

**For this demo, we will be using Nomic’s embedding model and MongoDB in order to accomplish this**

[**Nomic AI**](https://home.nomic.ai/) builds tools to enable anyone to interact with AI scale datasets and models. Nomic [Atlas](https://blog.nomic.ai/posts/atlas.nomic.ai) enables anyone to instantly visualize, structure, and derive insights from millions of unstructured data points. The text embedder,  known as Nomic Embed, is the backbone of Nomic Atlas, allowing users to search and explore their data in new ways.

<!--truncate-->

[**MongoDB**](https://www.mongodb.com/) is a popular open-source NoSQL database that is known for its flexibility, scalability, and performance. Unlike traditional relational databases that store data in tables with fixed schemas, MongoDB uses a document-oriented approach to store data in flexible, JSON-like documents. [**MongoDB Atlas**](https://www.mongodb.com/atlas/database) is a database service that drastically simplifies how people can build AI-enriched applications. It helps reduce complexity by allowing for low-latency deployments worldwide, automated scaling of compute and storage resources, and a unified query API, integrating operational, analytical, and vector search data services.

However, one thing still needs to be solved: how to generate and store embeddings of your existing data on the fly.


**SuperDuperDB helps to bridge this gap.**

[**SuperDuperDB**](https://superduperdb.com/) is a Python framework that directly integrates AI models, APIs, and vector search engines with existing databases. 
With SuperDuperDB, one can:

* Seamlessly integrate popular embedding APIs and open-source models with your database, enabling automatic generation of embeddings for your existing data in your database 

* Effortlessly manage different embedding models and APIs ( text, images) to suit diverse needs.

* Empower your database to process new queries instantly and create embeddings in real time.

**One bonus point** about using SuperDuperDB is its flexibility in integrating with custom functions. 

**We would use this to integrate with the Nomic embedding model.**



### **Let’s integrate the Nomic embedding model into your MongoDB using SuperDuperDB.**



**Step 1: Install SuperDuperDB and Nomic**
```python
    pip install superduperdb 
    pip install nomic
```


**Step 2: Set your NOMIC API Key**

Get your Nomic API key from the [Nomic Atlas website](https://atlas.nomic.ai/) and add it as an environment variable
```python
    import nomic 
    NOMIC_API_KEY = "<YOUR-NOMIC_API_KEY>"
    nomic.cli.login(NOMIC_API_KEY)
```


**Step 3: Connect SuperDuperDB to your MongoDB database and define the collection**

*For this demo, we will be using the default MongoDB testing connection. However, this can be switched to a local MongoDB instance, MongoDB with authentication and even MongoDB Atlas URI.*
```python
    from superduperdb import superduper
    from superduperdb.backends.mongodb import Collection
    
    mongodb_uri = "mongomock://test"
    artifact_store = 'filesystem://./my'
    
    
    db = superduper(mongodb_uri, artifact_store=artifact_store)
    
    my_collection = Collection("documents")
 ```   

*Note that*

* *The MongoDB URI can also be either local-hosted or MongoDB Atlas*

* *We also defined an artefact store path locally to store the model and data artefacts*

Next, let’s ingest some sample data directly
```python
    from superduperdb import Document
    
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
    
    
    
    db.execute(my_collection.insert_many([Document(r) for r in data]))
```
*If you already have existing data in your collection, skip the step above*

Next, view the first row of your collection 
```python
    result = db.execute(my_collection.find_one())
    print(result)
```


**Step 4: Define the embedding model in a function wrapper**
```python
    from superduperdb import Model, vector
    from nomic import embed
    
    
    def generate_embeddings(input:str) -> list[float]:
        """Generate embeddings from Nomic Embedding API.
    
        Args:
            input_text: string input.
        Returns:
            a list of embeddings. Each element corresponds to the each input text.
        """
       
        outputs = embed.text(texts=[input], model='nomic-embed-text-v1')
        return outputs["embeddings"][0]
    
    
    model = Model(identifier='nomic_embedding_model', object=generate_embeddings, encoder=vector(shape=(768,)))
```
*Note:*

* *The model identifier is a name to identify the model. It can be any name of choice*

* *The object is to call the custom function  created to generate the nomic embeddings*

* *The encoder argument above denotes the shape of the expected embedding output. The expected embedding shape for the NOMIC model is 768*



Test your model output **on the fly**. 
```python
    model.predict("This is a test")
```


**Step 5: Add the Nomic Embed model and the vector index to the database**
```python
    from superduperdb import Listener, VectorIndex
    
    collection_field_name = "description"
    
    
    listener = Listener(model=model,key=collection_field_name,select=my_collection.find())
    
    # define the vector index
    db.add(VectorIndex(identifier=f'my-mongodb-{model.identifier}', indexing_listener= listener))
```
*The **Listener Class** is used to listen to user queries and convert the queries to vectors automatically using the embedding model. More information about Listener can be found[ here](https://docs.superduperdb.com/docs/docs/fundamentals/component_abstraction).*

At this stage, an embedding of all the rows in the description column is automatically populated.



### **Congratulations, you are ready to run a vector search!**

Run a simple vector search query with the code below

*For this demo, we would limit the search results to 2. Feel free to increase the limit*
```python
    user_query = 'sport articles'
    limit_search_results = 2
    
    
    result = db.execute(
        my_collection
            .like(Document({'description': user_query}), vector_index=f'my-mongodb-{model.identifier}', n=limit_search_results)
            .find({}, {'title': 1, 'description': 1, 'score': 1})
    )
```
To view the result
```python
    for r in result:
      print(r.unpack())
```


### Conclusion

Now that you have implemented the vector search functionality, the next step  would be to add a chat model to the system to create a complete, simple RAG system. This can also be done with SuperDuperDB. 

To explore more, check out our other use cases in the [documentation](https://docs.superduperdb.com/docs/category/use-cases)

**SuperDuperDB also supports vector-supported SQL databases like Postgres and non-vector-supported databases like MySQL and DuckDB.**

Please check the [*documentation*](https://docs.superduperdb.com/docs/category/data-integrations) for more information about its functionality and the range of data integrations it supports.

### Useful Links

- **[Website](https://superduperdb.com/)**
- **[GitHub](https://github.com/SuperDuperDB/superduperdb)**
- **[Documentation](https://docs.superduperdb.com/docs/category/get-started)**
- **[Blog](https://docs.superduperdb.com/blog)**
- **[Example Use Cases & Apps](https://docs.superduperdb.com/docs/category/use-cases)**
- **[Slack Community](https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA)**
- **[LinkedIn](https://www.linkedin.com/company/superduperdb/)**
- **[Twitter](https://twitter.com/superduperdb)**
- **[Youtube](https://www.youtube.com/@superduperdb)**

### Contributors are welcome!

SuperDuperDB is open-source and permissively licensed under the [Apache 2.0 license](https://github.com/SuperDuperDB/superduperdb/blob/main/LICENSE). We would like to encourage developers interested in open-source development to contribute in our discussion forums, issue boards and by making their own pull requests. We'll see you on [GitHub](https://github.com/SuperDuperDB/superduperdb)!
