<p align="center">
  <a href="https://www.superduperdb.com">
    <img width="90%" src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/img/SuperDuperDB_logo_color.svg">
  </a>
</p>
<div align="center">

# Combine AI with your favorite database! 

</div>

<div align="center">

## <a href="https://superduperdb.github.io/superduperdb/"><strong>Docs</strong></a> | <a href="https://docs.superduperdb.com/blog"><strong>Blog</strong></a> | <a href="https://docs.superduperdb.com/docs/category/use-cases"><strong>Use-Cases</strong></a> | <a href="https://demo.superduperdb.com"><strong> Live Notebooks</strong></a> | <a href="https://github.com/SuperDuperDB/superduper-community-apps"><strong>Community Apps</strong></a> |  <a href="https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA"><strong> Slack </strong></a> | <a href="https://www.youtube.com/channel/UC-clq9x8EGtQc6MHW0GF73g"><strong> Youtube </strong></a>

</div>


<div align="center">
	<a href="https://pypi.org/project/superduperdb"><img src="https://img.shields.io/pypi/v/superduperdb?color=%23007ec6&label=pypi%20package" alt="Package version"></a>
	<a href="https://pepy.tech/project/superduperdb"><img src="https://pepy.tech/badge/superduperdb" alt="Downloads"></a>
  	<a href="https://pypi.org/project/superduperdb"><img src="https://img.shields.io/pypi/pyversions/superduperdb.svg" alt="Supported Python versions"></a>    
	<a href="https://github.com/SuperDuperDB/superduperdb/actions/workflows/ci_code.yml"><img src="https://github.com/SuperDuperDB/superduperdb/actions/workflows/ci_code.yml/badge.svg?branch=main" /></a>	
	<a href="https://codecov.io/gh/superduperdb/superduperdb/branch/main"><img src="https://codecov.io/gh/superduperdb/superduperdb/branch/main/graph/badge.svg" alt="Coverage"></a>
	<a href="https://github.com/superduperdb/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-green" alt="License - Apache 2.0"></a>	
	<a href="https://twitter.com/superduperdb" target="_blank"><img src="https://img.shields.io/twitter/follow/nestframework.svg?style=social&label=Follow @SuperDuperDB"></a>

<b>English</b> |
[**中文**](README_zh.md) 



</div>

---
`SuperDuperDB is open-source: Please leave a star to support the project! ⭐`



# What is SuperDuperDB?

SuperDuperDB is an open-source framework that effortlessly combines AI with your preferred database.


Rather than operating as an independent database, SuperDuperDB introduces layers of functionality, interfaces, and integration points that enhance the native capabilities of your database. It prepares your database to support various AI applications, including

* Generative AI & LLM-Chat
* Vector Search
* Standard Machine Learning Use-Cases (Classification, Segmentation, Recommendation, etc.)
* Highly customized AI use-cases 

By doing so, SuperDuperDB aims to reduce the dependency on complex MLOps pipelines or specialized vector databases, and eliminate the operation costs associated with them. 


# How does it work?

SuperDuperDB provides a client-side library that transparently wires together critical components involved in AI/Data integration. That includes:

* Compatible Access for NoSQL and SQL databases.
* Interfacing with popular AI frameworks and APIs.
* Interoperability with scalable compute frameworks.
* Flexible data types including video, images and audio.
* Real-time data integration and streaming inference.
* Vector Embeddings, Indexing and Searching.
* Model chaining for inference and training purposes.
* Movel Versioning, Lineage tracking, and  Rollback.

This adaptable data layer can be customized to meet specific application requirements using a declarative API, robust enough to construct AI workflows ranging from simple ML tasks to entire AI applications like LLM-Chat.


<p align="center">
  <img width="100%" src="docs/hr/static/img/superduperdb.gif">
</p>



## Why is it needed ?
When building a modern AI application, you don’t want to worry about how are being brought to your model, or what happens with the outputs.

Addressing this issue typically involves adopting complex approaches like MLops or utilizing specialized databases for vector searching. However, these solutions introduce complications such as data duplications and inconsistencies across systems, increasing costs due to data movement overheads and posing security risks as data becomes dispersed.

SuperDuperDB is here to help you reduce development time and operational costs of AI applications, by transforming your database into an AI powerhouse.

SuperDuperDB does the following for you:

1. It streamlines data-related challenges, such as ETL, feature shaping, and similarity searching.
2. It works directly with images, video, audio in your database, and any type which can be encoded as bytes in Python.
3. It compute outputs automatically and immediately as new data arrives, enabling streaming inference and real-time applications.
4. It aligns computational and data access optimizations for scalable model training.
5. It provides a declarative API for building AI workflows, ranging from straightforward ML tasks to complex AI apps like LLM-Chat.
6. It integrates any function, program, script or algorithm from the Python ecosystem to enhance your workflows.
7. It streamlines the development, deployment, monitoring, and maintenance of AI workflows.


For more information about SuperDuperDB and why we believe it is much needed, [read this blog post](https://docs.superduperdb.com/blog/superduperdb-the-open-source-framework-for-bringing-ai-to-your-datastore/) or watch the video:

<div align="center">	
	
[![SuperDuperDB in a Nutshell](https://img.youtube.com/vi/FxJs7pbHj3Q/0.jpg)](https://www.youtube.com/watch?v=FxJs7pbHj3Q)

</div>




## Integrated Technologies

### AI Frameworks and Models:

<table>
    <tr>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/frameworks/2Xpytorch.png" width="139px"/>
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/frameworks/2Xscikit-learn.png" width="139px"/>
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/frameworks/2Xhuggingface-transformers.png" width="139px"/>
        </td>
    </tr>
</table>



### Pre-Integrated AI APIs:

<table >
    <tr>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/apis/2Xopenai.png" width="139px"/>
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/apis/2Xcohere.png" width="139px"/>
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/apis/2Xanthropic.png" width="139px"/>
        </td>
    </tr>
</table>


### Datastores

|   Backend   | Data | Metadata | Artifacts | Require Changes |  
|:-----------:|------|----------|-----------|-----------------|
|   MongoDB   | X    | X        | X         | No              |
| Mongo Atlas | X    | X        | X         | No              |
| PostgreSQL  | X    |          | X         | No              |
|    MySQL    | X    | X        | X         | Yes             |
|   SQLite    | X    | X        | X         | No              |
|   DuckDB    | X    | X        | X         | No              |
|  SnowFlake  | X    |          |           | No              |
|  BigQuery   | X    |          |           | No              |
| ClickHouse  | X    |          |           | No              |

With more options coming soon ...

<table>
    <tr>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xdatafusion.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xdruid.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Ximpala.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xmssql.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xoracle.png" width="139px" />
        </td>
    </tr>
    <tr>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xpandas.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xpolars.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xpyspark.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xtrino.png" width="139px" />
        </td>
        <td align="center" width="140" height="112.43">
            <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/databases/2Xaws-s3.png" width="139px" />
        </td>
    </tr>

</table>



## Usage
 
You can install superduperdb directly on your application:

```
pip install superduperdb
```

Or, you can test our demo applications on your local machine: 
```
docker run -p 8888:8888 superduperdb/demo:latest
```

Or even better, you can try them [live on your browser](https://demo.superduperdb.com).

Here is a list of available notebooks to get started.

<table>
  <tr>
    <td width="30%">
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/multimodal_image_search_clip.ipynb">
        <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/featured-examples/image-search.svg" />
      </a>
    </td>
    <td width="30%">
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/video_search.ipynb">
        <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/featured-examples/video-search.svg" />
      </a>
    </td>
    <td width="30%">
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/question_the_docs.ipynb">
        <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/featured-examples/semantic-search.svg" />
      </a>
    </td>
  </tr>
  <tr>
    <th>
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/multimodal_image_search_clip.ipynb">Text-To-Image Search</a>
    </th>
    <th>
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/video_search.ipynb">Text-To-Video Search</a>
    </th>
    <th>
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/question_the_docs.ipynb">Question the Docs</a>
    </th>
  </tr>
  <tr>     
    <td width="30%">
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/vector_search.ipynb">
        <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/featured-examples/document-search.svg" />
      </a>
    </td>
    <td width="30%">
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/mnist_torch.ipynb">
        <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/featured-examples/machine-learning.svg" />
      </a>
    </td>
    <td width="30%">
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/transfer_learning.ipynb">
        <img src="https://raw.githubusercontent.com/SuperDuperDB/superduperdb/main/docs/hr/static/icons/featured-examples/transfer-learning.svg" />
      </a>
    </td>
  </tr>
  <tr>
    <th>
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/vector_search.ipynb">Semantic Search Engine</a>
    </th>
    <th>
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/mnist_torch.ipynb">Classical Machine Learning</a>
    </th>
    <th>
      <a href="https://demo.superduperdb.com/user-redirect/lab/tree/transfer_learning.ipynb">Cross-Framework Transfer Learning</a>
    </th>
  </tr>
</table>

Also find use-cases and apps built by the community in the [superduper-community-apps repository](https://github.com/SuperDuperDB/superduper-community-apps).






## Preview

The following snippets will give you an idea of how to Integrate, Train and Manage any AI model (whether from open-source, commercial models or self-developed) using SuperDuperDB:


- Install and deploy model:

```
m = db.add(
    <sklearn_model>|<torch_module>|<transformers_pipeline>|<arbitrary_callable>,
    preprocess=<your_preprocess_callable>,
    postprocess=<your_postprocess_callable>,
    encoder=<your_datatype>
)
```

- Predict:

```
m.predict(X='<input_column>', db=db, select=<mongodb_query>, listen=False|True, create_vector_index=False|True)
```

- Train model:

```
m.fit(X='<input_column_or_key>', y='<target_column_or_key>', db=db, select=<mongodb_query>|<ibis_query>)
```


- Integrate pre-trained models from AI services. 

```
m = db.add(
    OpenAI<Task>|Cohere<Task>|Anthropic<Task>(*args, **kwargs),   # <Task> - Embedding,ChatCompletion,...
)
```


#### - Integrate AI with your database ([see the docs](https://docs.superduperdb.com/docs/docs/walkthrough/apply_models)) 

Integrate any AI models and APIs with your databases, without the need for additional pre-processing steps, ETL or boilerplate code.


```python
import pymongo
from sklearn.svm import SVC

from superduperdb import superduper

# Make your db superduper!
db = superduper(pymongo.MongoClient().my_db)

# Models client can be converted to SuperDuperDB objects with a simple wrapper.
model = superduper(SVC())

# Add the model into the database
db.add(model)

# Predict on the selected data.
model.predict(X='input_col', db=db, select=Collection(name='test_documents').find({'_fold': 'valid'}))
```



#### - Scalable Model Training ([see the docs](https://docs.superduperdb.com/docs/docs/walkthrough/training_models))

Train AI models on large, diverse datasets simply by querying your training data, without additional ingestion and pre-processing.  Ensured optimal performance via in-build computational optimizations.


```python
import pymongo
from sklearn.svm import SVC

from superduperdb import superduper

# Make your db superduper!
db = superduper(pymongo.MongoClient().my_db)

# Models client can be converted to SuperDuperDB objects with a simple wrapper.
model = superduper(SVC())

# Predict on the selected data.
model.train(X='input_col', y='target_col', db=db, select=Collection(name='test_documents').find({'_fold': 'valid'}))
```



#### - Vector-Search your data ([see the docs](https://docs.superduperdb.com/docs/docs/walkthrough/vector_search)

No need to duplicate and migrate your data to additional specialized vector databases - turn your existing battle-tested database into a fully-fledged multi-modal vector-search database, including easy generation of vector embeddings and vector indexes of your data with preferred models and APIs.


```python
# First a "Listener" makes sure vectors stay up-to-date
indexing_listener = Listener(model=OpenAIEmbedding(), key='text', select=collection.find())

# This "Listener" is linked with a "VectorIndex"
db.add(VectorIndex('my-index', indexing_listener=indexing_listener))

# The "VectorIndex" may be used to search data. Items to be searched against are passed
# to the registered model and vectorized. No additional app layer is required.
db.execute(collection.like({'text': 'clothing item'}, 'my-index').find({'brand': 'Nike'}))
```

#### - Integrate AI APIs to work together with other models ([see the docs](https://docs.superduperdb.com/docs/docs/fundamentals/procedural_vs_declarative_api))

Add and leverage any function, program, script or algorithm from the Python ecosystem to enhance your workflows and applications. Drill down to any layer of implementation, including to the inner workings of your models while operating SuperDuperDB with simple Python commands.
Use OpenAI, PyTorch or Hugging face model as an embedding model for vector search.

```python
# Create a ``VectorIndex`` instance with indexing listener as OpenAIEmbedding and add it to the database.
db.add(
    VectorIndex(
        identifier='my-index',
        indexing_listener=Listener(
            model=OpenAIEmbedding(identifier='text-embedding-ada-002'),
            key='abstract',
            select=Collection(name='wikipedia').find(),
        ),
    )
)
# The above also executes the embedding model (openai) with the select query on the key.

# Now we can use the vector-index to search via meaning through the wikipedia abstracts
cur = db.execute(
    Collection(name='wikipedia')
        .like({'abstract': 'philosophers'}, n=10, vector_index='my-index')
)
```




- **[Streaming Inference](https://docs.superduperdb.com/docs/docs/walkthrough/daemonizing_models_with_listeners):** Have your models compute outputs automatically and immediately as new data arrives, keeping your deployment always up-to-date.
- **[Model Chaining](https://docs.superduperdb.com/docs/docs/walkthrough/linking_interdependent_models/)**: Easily setup complex workflows by connecting models and APIs to work together in an interdependent and sequential manner.
- **[Difficult Data-Types](https://docs.superduperdb.com/docs/docs/walkthrough/encoding_special_data_types/)**: Work directly with images, video, audio in your database, and any type which can be encoded as `bytes` in Python.
- **[Feature Storing](https://docs.superduperdb.com/docs/docs/walkthrough/encoding_special_data_types):** Turn your database into a centralized repository for storing and managing inputs and outputs of AI models of arbitrary data-types, making them available in a structured format and known environment.



#### - Use Encoders for arbitrary Data Types!:

Add a Llama 2 model to SuperDuperDB

```python
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = Pipeline(
    identifier='my-sentiment-analysis',
    task='text-generation',
    preprocess=tokenizer,
    object=pipeline,
    torch_dtype=torch.float16,
    device_map="auto",
)

# You can easily predict on your collection documents.
model.predict(
    X=Collection(name='test_documents').find(),
    db=db,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200
)
```

#### - Use models outputs as inputs to downstream models:

```python
model.predict(
    X='input_col',
    db=db,
    select=coll.find().featurize({'X': '<upstream-model-id>'}),  # already registered upstream model-id
    listen=True,
)
```



## Community & Getting Help 

#### If you have any problems, questions, comments, or ideas:
- Join <a href="https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA">our Slack</a> (we look forward to seeing you there).
- Search through <a href="https://github.com/SuperDuperDB/superduperdb/discussions">our GitHub Discussions</a>, or <a href="https://github.com/SuperDuperDB/superduperdb/discussions/new/choose">add a new question</a>.
- Comment <a href="https://github.com/SuperDuperDB/superduperdb/issues/">an existing issue</a> or create <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">a new one</a>.
- Help us to improve SuperDuperDB by providing your valuable feedback <a href="https://docs.google.com/forms/d/e/1FAIpQLScKNyLCjSEWAwc-THjC7NJVDNRxQmVR5ey30VVayPhWOIhy1Q/viewform">here</a>!
- Email us at `gethelp@superduperdb.com`.
- Feel free to contact a maintainer or community volunteer directly! 



## Contributing  

#### There are many ways to contribute, and they are not limited to writing code. We welcome all contributions such as:


- <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">Bug reports</a>
- <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">Documentation improvements</a>
- <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">Enhancement suggestions</a>
- <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">Feature requests</a>
- <a href="https://github.com/SuperDuperDB/superduperdb/issues/new/choose">Expanding the tutorials and use case examples</a>

Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Contributors
#### Thanks goes to these wonderful people:

<a href="https://github.com/SuperDuperDB/superduperdb/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SuperDuperDB/superduperdb" />
</a>


## License  

SuperDuperDB is open-source and intended to be a community effort, and it wouldn't be possible without your support and enthusiasm.
It is distributed under the terms of the Apache 2.0 license. Any contribution made to this project will be subject to the same provisions.

## Join Us 

We are looking for nice people who are invested in the problem we are trying to solve to join us full-time. Find roles that we are trying to fill <a href="https://join.com/companies/superduperdb">here</a>!
