SuperDuperDB Documentation
==========================

SuperDuperDB is:

* An AI-database management system
    The AI data/ model workflow is highly complex with multiple interdependent steps where
    models and data repeatedly interface with one another.
    SuperDuperDB is exactly the right environment to bring the data and AI worlds together.
* Shares all of the standard advantages of a database
    * Access control
    * User groups
    * Security
    * Scalability
    * Indexing
* Adds novel features to databases which are not possible without AI
    * Arbitrary data imputations
    * Semantic indexing based on linear algebra
    * Associating models directly with the data which trained them for transparent model lineage.
* Integrates PyTorch models natively to the database
    It allows users to work intuitively and efficiently with AI.
* Is ready for the latest generation of AI models
    New developments in AI imply that architectures such as `Retro <https://www.deepmind.com/publications/improving-language-models-by-retrieving-from-trillions-of-tokens>`_ and will
    need ready access to your data. SuperDuperDB allows users to build models which reference
    the database explicitly.
* Is fully modular and allows users to
    * provide arbitrary PyTorch models
    * apply arbitrary pre- and post-processing
    * define custom data types, e.g. tensors, images, audio, video
    * train models
* Allows you to train models directly on the database
    SuperDuperDB has prebaked support for training semantic lookup tasks and imputation style
    learning tasks. User-defined training setups may also be
    supplied and stored in the database. The technology leverages data loaders which enable direct
    cross talk between MongoDB and PyTorch.

Getting started
---------------

.. code-block:: python
    :caption: In :code:`my_module.py`

    class FloatTensor:
        @staticmethod
        def encode(x):
            x = x.numpy()
            assert x.dtype == numpy.float32
            return memoryview(x).tobytes()

        @staticmethod
        def decode(bytes_):
            array = numpy.frombuffer(bytes_, dtype=numpy.float32)
            return torch.from_numpy(array).type(torch.float)


.. code-block:: python
    :caption: Interactive session

    >>> from superduperdb.client import SuperDuperClient
    >>> from my_module import FloatTensor
    >>> import torch
    >>> docs = SuperDuperClient().my_database.my_collection
    >>> docs.create_converter('float_tensor', FloatTensor())
    >>> docs.create_model('linear_embedding', torch.nn.Linear(1024, 64), key='x')
    >>> docs.insert_many([{'x': {'_content': {'bytes': FloatTensor.encode(torch.randn(1024))}}}
                          for _ in range(100)])
    <pymongo.results.InsertManyResult at 0x15bb2b100>
    >>> docs.find_one()
    {'_id': ObjectId('63bbe91425c3c66430781968'),
     'x': tensor([ 0.4183,  0.8675, -1.1050,  ..., -1.1262,  1.1444, -1.6189]),
     '_fold': 'train',
     '_outputs': {'x': {'linear_embedding': tensor([ 0.7209,  0.7174,  0.7313, -0.4618,  0.4003, -0.6236, -0.3384, -0.6447,
               -0.4203,  0.1753,  0.3884, -0.5631, -0.3746, -0.1693,  0.0548, -0.2126,
               -0.5200,  0.2948,  0.1011, -0.3713, -0.3350,  1.0404,  0.6277, -0.2002,
               -0.8467, -0.1751,  0.4663,  0.4029, -0.3137, -0.8392, -0.1933,  0.3132,
                0.1859,  0.1336,  0.0895, -0.0495, -0.0224,  0.2773, -0.2423, -0.1698,
               -1.1780, -0.3219, -0.7944, -0.0969, -0.1691,  0.3163,  0.0658,  0.4155,
               -1.1576,  0.3640,  0.2191, -0.6726,  0.3572,  1.3214, -0.1269,  0.5001,
                0.0653,  0.6070, -0.0184, -0.4811,  0.2756, -0.0257, -0.5821,  0.7546])}}}

.. autoclass:: superduperdb.collection.Collection
  :members:

.. autoclass:: superduperdb.database.Database
  :members:

.. autoclass:: superduperdb.client.SuperDuperClient
  :members:




