Minimum working example
=======================

Try this example to check everything is working as expected. For an explanation of the concepts
and detailed usage of SuperDuperDB see :ref:`here <SuperDuperDB Concepts>`.

.. code-block:: python
    :caption: In :code:`my_module.py`

    import numpy
    import torch

    class FloatTensor:
        types = (torch.FloatTensor, torch.Tensor)

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
    >>> docs.create_type('float_tensor', FloatTensor())
    >>> docs.create_model('linear_embedding', torch.nn.linear(1024, 64))
    >>> docs.create_watcher('linear_embedding', 'x')
    >>> docs.insert_many([{'x': torch.randn(1024) for _ in range(100)])
    <pymongo.results.InsertManyResult at 0x15bb2b100>
    # watch a bit ...
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
