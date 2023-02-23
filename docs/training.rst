********************************
Model training with SuperDuperDB
********************************

There are currently two ways to train a model in SuperDuperDB.
The first way uses ``superduperdb.collection.Collection.create_semantic_index``
and the second way uses ``superduperdb.collection.Collection.create_imputation``.
These are very generic wrappers for respectively representation training, and tasks
which involving predicting something from something else.

++++++++++
Components
++++++++++

Training your own semantic index requires a few more components than using pre-trained models.
For training we need these things:

- One or more **models**
- A **loss** function
- One or more **metrics**
- ...

.. code-block:: python
    :caption: Semantic index training
    >>> superduperdb.client import SuperDuperClient
    >>> docs = SuperDuperClient().my_database.my_collection
    >>> from my_codebase.models import QueryEncoder
    >>> from my_codebase.models import DocumentEncoder
    >>> from my_codebase.losses import representation_loss
    >>> ...
    >>> docs.create_model('query_encoder', QueryEncoder(), active=False)
    >>> docs.create_model('document_encoder', DocumentEncoder(), filter=my_mongo_query, active=True)

    >>> docs.create_semantic_index('my_index', ('query_encoder', 'document_encoder'), metrics=)

