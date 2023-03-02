*********************
SuperDuperDB Concepts
*********************

There are several key concepts which SuperDuperDB uses and implements in order to add the functionality 
necessary for deploying AI models with MongoDB. By integrating features based on these concepts, SuperDuperDB
allows users to work naturally with MongoDB collections together with PyTorch models.
These concepts are:

* :ref:`Types`
* :ref:`Content`
* :ref:`Models`
* :ref:`Watchers`
* :ref:`Semantic Indexes`
* :ref:`Imputations`
* :ref:`Jobs`

Types
=====

A type is a Python object registered with a SuperDuperDB collection which manages how
model outputs or database content are converted to and from ``bytes`` so that these may be
stored and retrieved from the database. Creating types is a prerequisite to adding models
which have non-Jsonable outputs to a collection, as well as adding content to the database
of a more sophisticated variety, such as images, tensors and so forth.

A type is any class with ``.encode`` and ``.decode`` methods
as well as an optional ``.types`` property.

Read in more detail :ref:`here <Types in SuperDuperDB>`.

Content
=======

In order to add data of a certain "type" to SuperDuperDB, one uses the concept of "content". This refers 
to a data point, whose "content" may be described in a number of ways - either explicitly by supplying the raw
bytes or implicitly, by pointing to a URL or bucket location which contains the content. For those bits of 
content which are referred to implicitly, SuperDuperDB creates a :ref:`job <Jobs>` which fetches the bytes from the 
described location, and inserts these into MongoDB.

Read in more detail :ref:`here <Adding interesting content to SuperDuperDB>`.

Models
======

A model in SuperDuperDB is a PyTorch model, with (optionally) two additional methods ``preprocess`` 
and ``postprocess``. These methods are necessary so that the model knows how to convert content from 
the database to tensors, and also to convert outputs of the object into a form which is appropriate 
to be saved in the database.

Read in more detail :ref:`here <Models - an extension of PyTorch models>`

Watchers
========

Once you have one or more models registered with SuperDuperDB, the model(s) can be set up to 
listen to certain sub-keys (``key``) or full documents in MongoDB collections, and to compute outputs
over those items when new data comes in or updates are made to the database.

When a watcher is created based on a SuperDuperDB model, a dataloader is created which 
loads data from the database, passes the data inside the configured ``key`` to the model's
``preprocess`` method, batches the tensors and passes these to the model's ``forward`` method,
and finally unpacks the batch and applies the model's ``postprocess`` method to the lines of 
output from the model. The results are saved in ``"_outputs.<key>.<model_name>"`` of the collection 
documents.

Read in more detail :ref:`here <Watchers in SuperDuperDB>`.

Semantic Indexes
================

Models and their outputs may be used in concert, to make the content of SuperDuperDB collections
searchable. A **semantic index** is one or more models, which produce PyTorch vector or tensor 
outputs. A semantic index may be leveraged using the ``Collection.find`` or ``Collection.find_one``
method, for finding matches based on the individual models which are registered with the semantic index.

Examples of the use of semantic indexes are:

* Search by meaning in NLP
* Similar image recommendation
* Similar document embedding and recommendation
* Facial recognition
* ... (there are many possibilities)

For example, a semantic index could consist of a pair of models, where one model understands text and
the other understands images. Using this pair, one can search for images using a textual description of the image.

SuperDuperDB may be used to train or fine-tune semantic-indexes.

Read in more detail :ref:`here <Semantic indexes for flexibly searching data>`.

Imputations
===========

An imputation is a pair of models, where one model is used to predict the output of the other model.
This subsumes many use cases:

* Classification
* Regression
* Autoregressive modelling (language modelling, time-series modelling, ...)
* Generative adversarial learning
* Image segmentation
* Bounding box regression
* ... (there are many possibilities)

Read in more detail :ref:`here <Imputations for filling in data>`.

Jobs
====

Whenever SuperDuperDB does any of the following:

* Data insertion
* Data updates
* Model creation
* Model training
* Semantic index updates
* Neighbourhood calculations

then SuperDuperDB is required to perform certain longer running computations.
These computations are wrapped as "jobs" and the jobs are carried out asynchronously on
a pool of parallel workers.

SuperDuperDB may also be used in the foreground, so that calculations block the Python program. 
This is recommended for development purposes only.

Read in more detail :ref:`here <Jobs - scheduling of training and model outputs>`.
