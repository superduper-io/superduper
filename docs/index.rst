**************************
SuperDuperDB Documentation
**************************

+++++++++++++++++++++
What is SuperDuperDB?
+++++++++++++++++++++

The SuperDuperDB project aims to provide a unified environment for open-source AI model development
and production together with PyTorch and MongoDB.

That means:

* SuperDuperDB allows users to integrate PyTorch models into MongoDB.
* Arbitrary PyTorch models can be configured to save their outputs directly in the database alongside the inputs used to compute them.
* Model outputs can be used in database queries, for example, enabling queries which use linear algebra.
* SuperDuperDB can handle models with outputs of various types - anything which can be converted from a Python object to raw bytes and back.
* SuperDuperDB allows users to train their models directly with the database, without the need for intermediate data caching or preparation steps.

+++++++++++++++++++++++++++
What is SuperDuperDB not?
+++++++++++++++++++++++++++

* SuperDuperDB is **not** a database (despite the name) - it makes your database super-duper.
* SuperDuperDB doesn't perform AutoML (like MindsDB). We think it's good to offer users the opportunity to choose their own models.
* SuperDuperDB is not a pure vector search database - it leverages open-source vector search components, which allow certain types of data querying. However SuperDuperDB does not reinvent the wheel there.

+++++++++++++++++++++++++++++++++++++++++++++++++++
What types of use-cases does SuperDuperDB excel at?
+++++++++++++++++++++++++++++++++++++++++++++++++++

* If model outputs need to be kept up-to-date when new data comes in.
* Transfer learning - downstream models injest the outputs of other models as inputs.
* Managing multiple model variants and versions which potentially listen to different sets of data.
* Multimodal branching models which injest JSON type data points as inputs.
* AI involving data types not supported by standard data bases, such as images.

Documentation contents
======================

.. toctree::
    :maxdepth: 2

    getting_started
    concepts
    cluster
    models
    semantic_indexes
    training
    examples/index
    full_usage

