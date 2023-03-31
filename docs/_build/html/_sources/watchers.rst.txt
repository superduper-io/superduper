Watchers
========

Model callbacks for continuously populating model outputs
---------------------------------------------------------

Once you have created a model to work with SuperDuperDB using ``Collection.create_model``, it's possible to 
set up the model to react to changes in a collection's data using a **watcher**. To create a watcher, use the
``Collection.create_watcher`` command:

.. code-block:: python
	
	>>> docs.create_watcher(model='my_model', key='x', filter_={}, 
	...                     loader_kwargs={'batch_size': 100, 'num_workers': 10)
	# lots of output

When this command is called, a :ref:`job <Jobs - scheduling of training and model outputs>` is
created which iterates through all documents selected by the ``filter`` key-word
and updates the outputs to the sub-field under the key ``"_outputs.<key>.<model>"``.

Whenever new data are inserted or updates are made, if these fall under the query given by ``filter_`` then the model outputs
are computed on this new data and updated to ``"_outputs.<key>.<model>."``.

How watchers work under the hood
--------------------------------

Once a watcher has been created, the documents selected by the ``filter_`` parameter are wrapped in a
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

Chaining watchers by using features
-----------------------------------


Now that the outputs are saved in the documents, they can be used to "featurize" the same
field over which they were computed:

.. code-block:: python

    >>> docs.find_one(features={'img': 'my_module'}, {'_outputs': 0})
    {'_id': ObjectId('6387bc38477124958d0b97d9'),
     'title': 'BODYSUIT - Long sleeved top',
     'img': tensor([ 0.0064,  0.0055, -0.0140,  ...,  0.0120,  0.0084, -0.0253]),
     '_fold': 'train',
     '_outputs': {'title': {'my_module': tensor([ 0.0064,  0.0055, -0.0140,  ...,  0.0120,  0.0084, -0.0253])}}}

You can see that the model outputs for ``my_module`` have been substituted into the ``img`` field.

Often watchers will depend on other watchers, if the output of one watcher is needed as the input for another
watcher. The way to do this is to set the ``features`` key-word when creating a watcher. This means
that when data is fetched from the database during computation of model outputs, the keys specified
in the ``features`` dictionary are replaced by outputs of the models given in the values.

.. code-block:: python

    >>> docs.create_watcher('other_model', 'img', filter={'img': {'$exists': 1}},
    ...                     features={'img': 'my_model')

This is a very useful feature, for instance, in transfer learning.

CNN watcher
-----------

Continuing :ref:`this example<CNN example>` we chain a featurizing computer vision model, with a linear classifier:

.. code-block:: python

    >>> from my_packages.models import CNN
    >>> docs.create_watcher(model='resnet', key='img', filter={'img': {'$exists': 1}})
    >>> docs.create_watcher('visual_classifier', key='img',
    ...                     filter={'img': {'$exists': 1}}, features={'img': 'resnet'})
    # wait a bit...
    >>> docs.find_one()
    {'_id': ObjectId('6387bc38477124958d0b97d9'),
     'img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=250x361>,
     '_outputs': {'img': {'resnet': tensor([0.0064,  0.0055, -0.0140,  ...,  0.0120,  0.0084, -0.0253])},
                          'visual_classifier': 'dark-lighting'}}


The ``create_watcher`` command applies the model to all of the documents which are selected by the ``filter``
parameter (default ``{}`` - all). The second watcher depends for its input features on the first
model. This is configured via the ``features={...}`` key-word. The fields in the dictionary
are substituted with the model-outputs defined there.
