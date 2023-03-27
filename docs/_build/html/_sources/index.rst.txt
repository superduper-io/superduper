**************************
SuperDuperDB Documentation
**************************

+++++++++++++++++++++
What is SuperDuperDB?
+++++++++++++++++++++

SuperDuperDB is a project which aims to facilitate building and integrating AI models together with data.
It does this by providing an open-source AI model development
and production environment together with PyTorch and MongoDB.

* SuperDuperDB allows users to integrate PyTorch models into MongoDB.
* Arbitrary PyTorch models can be configured to save their outputs directly in the database alongside the inputs used to compute them.
* Model outputs can be used in database queries, for example, enabling queries which use linear algebra.
* SuperDuperDB can handle models with outputs of various types - anything which can be converted from a Python object to raw bytes and back.
* SuperDuperDB allows users to train their models directly with the database, without the need for intermediate data caching or preparation steps.

+++++++++++++++++++++++++++
What is SuperDuperDB not?
+++++++++++++++++++++++++++

* SuperDuperDB is **not** a database (despite the name) - it *makes* your database *super-duper*.
* SuperDuperDB doesn't perform AutoML. We think it's good to offer users the opportunity to choose their own models.
* SuperDuperDB is not a pure vector search database - it leverages open-source vector search components, which allow certain types of data querying. However SuperDuperDB does not reinvent the wheel there.
* SuperDuperDB does not try to hack or create dialects of query languages such as the MongoDB query API or SQL. These languages do their job well already.

+++++++++++++++++++++++++++++++++++++++++++++++++++
What types of use-cases is SuperDuperDB suited for?
+++++++++++++++++++++++++++++++++++++++++++++++++++

* If model outputs need to be kept up-to-date when new data comes in.
* If training and/ or deployment data already sits in a MongoDB cluster.
* Transfer learning - downstream models injest the outputs of other models as inputs.
* Managing multiple model variants and versions which potentially listen to different sets of data.
* Multimodal branching models which injest JSON type data points as inputs.
* AI involving data types not supported by standard data bases, such as images.

++++++++++++++++++++++++++++++++++++++++++
How does SuperDuperDB differ from MindsDB?
++++++++++++++++++++++++++++++++++++++++++

MindsDB is an `open source project <https://github.com/mindsdb/mindsdb>`_ which offers in-database machine learning.

**SuperDuperDB can do these things which MindsDB can't**

* Users can define arbitrary data types for their documents using Python.
* As a consequence SuperDuperDB can easily support image data types (for example).
* SuperDuperDB allows users to work with PyTorch models natively.
* SuperDuperDB allows users to work with MongoDB natively.
* SuperDuperDB has a Python first approach - add your models and data directly from Jupyter.
* SuperDuperDB has very flexible training wrappers, to define your own training procedures.

**MindsDB can do these things which SuperDuperDB can't**

* MindsDB supports AutoML. For large PyTorch AI models, this can get very expensive, so we have chosen not to add this support.
* MindsDB supports SQL as a priority. We may add support for structured data bases later.
* MindsDB offers a user interface. We believe that Python is the right way to do AI and data science, so don't offer a UI.
* MindsDB offers many data connectors. With SuperDuperDB, you can combine the Python ecosystem with SuperDuperDB for data injestion.

Documentation contents
======================

.. toctree::
    :maxdepth: 2

    getting_started
    minimum_working_example
    examples/index
    concepts
    types
    content
    models
    watchers
    semantic_indexes
    imputations
    jobs
    cluster
    full_usage

