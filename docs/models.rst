**********************************
Deep dive into SuperDuperDB models
**********************************

Models in SuperDuperDB extend the notion of PyTorch models
by including pre-processing and post-processing. These are necessary
since the data in the database aren't necessarily in tensor format
and the outputs which one might like to query in the database also
aren't necessarily tensors.

Models in SuperDuperDB are created using the
``superduperdb.collection.Collection.create_model`` command.

There are two alternative paths to creating models.
Either one supplies a ``torch.nn.Module`` object with methods:

* ``preprocess``
* ``forward``
* ``postprocess``

.. code-block:: python

    class MyModule(torch.nn.Module):
        def __init__(self, n_input, n_output):
            super().__init__()
            self.layer = torch.nn.Linear(n_input, n_output)
            ...

        def preprocess(self, *args, **kwargs):
            ...

        def forward(self, *args, **kwargs):
            ...

        def postprocess(self, *args, **kwargs):
            ...


And supplies this to the method:

.. code-block:: python

    >>> from my_package import MyModule
    >>> docs = the_client.my_database.my_collection
    >>> docs.create_model('my_model', object=MyModule())

An alternative is to define ``preprocess`` and ``postprocess`` functions
which are supplied separately to the ``torch.nn.Module`` part of the
model:

.. code-block:: python

    def preprocess(*args, **kwargs):
        ...

    def forward(*args, **kwargs):
        ...

    def postprocess(*args, **kwargs):
        ...

This approach has the advantage of the methods being able to share data from the class's
``__init__`` signature.

.. code-block:: python

    >>> from my_package import preprocess, postprocess
    >>> docs.create_model('my_model', preprocess={'my_preprocess': preprocess},
    ...                   forward=torch.nn.Linear(n_input, n_output),
    ...                   postprocess={'my_postprocess': postprocess})

This has the advantage of modularity as the pre- and postprocessing parts can be shared between
models more easily.

In direct analogy to ``pymongo.collection.Collection.create_index``, it's possible to supply
a filter (query) which selects which documents a model is applied to.

For example, suppose that our model requires a certain field ``img`` to exist, in order to be applicable.
Then the following command would do the trick, and would restrict the model to the documents
selected by the filter:

.. code-block:: python

    >>> from my_package import MyModule
    >>> docs = the_client.my_database.my_collection
    >>> docs.create_model('my_model', object=MyModule(), filter={'img': {'$exists': 1}})

If our model has particular outputs not supported directly by MongoDB, then it's necessary
to provide a "type" which handles the conversion to and from ``bytes``. That will allow the
outputs of the model to be saved to the database.

.. code-block:: python

    class MyType:
        types = (Type1, Type2, ...)

        def encode(self, x):
            ...
            return my_bytes_string

        def decode(self, my_bytes_string):
            ...
            return x

An instance of this type is no supplied to the ``create_model`` method.

.. code-block:: python

    >>> from my_package import MyModule, MyType
    >>> docs = the_client.my_database.my_collection
    >>> docs.create_type('my_type', MyType())
    >>> docs.create_model('my_model', object=MyModule(), filter={'img': {'$exists': 1}}, type='my_type')

Once a model has been created, the documents selected by the ``filter`` are wrapped in a
``torch.utils.data.DataLoader`` and outputs are computed. For this, the utility function
``superduperdb.models.utils.apply_model`` is applied.
The basic logic of this function is that the ``preprocess`` part is wrapped in a
``torch.utils.data.DataSet`` object and the outputs of this are batched together using a dataloader
and passed to the ``forward`` part of the model.
Finally the batched outputs of the ``forward`` part are unpacked, and the ``postprocess`` part
is applied to the "rows" of the batch.

Exactly what your batches of data will look like inside this process, is illustrated by the following
lines of code:

.. code-block:: python

    >>> import torch.utils.data, torch
    >>> datapoints = [
    ...   [{'a': {'b': 1}, 'c': 2}, [0, 0]] for _ in range(10)
    ... ]
    >>> dataloader = torch.utils.data.DataLoader(datapoints, batch_size=2)
    >>> for batch in dataloader:
    ...     print(batch)
    [{'a': {'b': tensor([1, 1])}, 'c': tensor([2, 2])}, tensor([[0, 0], [0, 0]])]
    [{'a': {'b': tensor([1, 1])}, 'c': tensor([2, 2])}, tensor([[0, 0], [0, 0]])]
    [{'a': {'b': tensor([1, 1])}, 'c': tensor([2, 2])}, tensor([[0, 0], [0, 0]])]
    [{'a': {'b': tensor([1, 1])}, 'c': tensor([2, 2])}, tensor([[0, 0], [0, 0]])]
    [{'a': {'b': tensor([1, 1])}, 'c': tensor([2, 2])}, tensor([[0, 0], [0, 0]])]

You can see the ``DataLoader`` class drilling down into the “leaf” nodes of the individual data points,
and batching at the level of those leaves. For nested MongoDB documents, this is rather convenient,
since the nested structure of the records may be easily handled in using the standard
``torch.utils.data.DataLoader``.

By default, the arguments of the ``preprocess`` part of a model, is always an entire MongoDB
record. Optionally, however, models can act on certain sub-documents or fields, by specifying
the ``key`` parameter in creating the model:

.. code-block:: python

    >>> docs.find_one()
    {'_id': ObjectId('6387bc38477124958d0b97d9'),
     'title': 'BODYSUIT - Long sleeved top',
     'img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=250x361>,
     '_fold': 'train'}
    >>> docs.create_model('my_model', object=MyModule(), filter={'img': {'$exists': 1}},
                          type='my_type', key='title')
    # wait a bit
    >>> docs.find_one()
    {'_id': ObjectId('6387bc38477124958d0b97d9'),
     'title': 'BODYSUIT - Long sleeved top',
     'img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=250x361>,
     '_fold': 'train',
     '_outputs': {'title': {'my_module': tensor([ 0.0064,  0.0055, -0.0140,  ...,  0.0120,  0.0084, -0.0253])}}}

You can see that the outputs of the model are saved in the ``_outputs.title.my_module`` field.
By specifying the ``key`` field, you avoid the necessity of having to delve into the depth
of the records inside your model, which makes your setup more flexible and easier to understand.

Now that the outputs are saved in the documents, they can be used to "featurize" the same
field over which they were computed:

.. code-block:: python

    >>> docs.find_one(features={'title': 'my_module'}, {'_outputs': 0})
    {'_id': ObjectId('6387bc38477124958d0b97d9'),
     'title': 'BODYSUIT - Long sleeved top',
     'img': tensor([ 0.0064,  0.0055, -0.0140,  ...,  0.0120,  0.0084, -0.0253]),
     '_fold': 'train',
     '_outputs': {'title': {'my_module': tensor([ 0.0064,  0.0055, -0.0140,  ...,  0.0120,  0.0084, -0.0253])}}}

You can see that the model outputs for ``my_module`` have been substituted into the ``img`` field.
This is a very useful feature, when models depend on one another, e.g. in transfer learning.