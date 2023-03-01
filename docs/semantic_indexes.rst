Semantic indexes for flexibly searching data
============================================

When you've created one or more models in SuperDuperDB which have tensor outputs,
SuperDuperDB may be used to search through the data, on the basis of input which
may be accepted by those models.

For example, continuing the example begun :ref:`here <CNN example>`, 
let's make a semantic index on the basis of the ``img`` field ,
utilizing the ``resnet`` model from before. We first need to create a measure function
which will be used to compare tensor outputs of the contained models:

.. code-block:: python

    def dot(x, y):
        return x.matmul(y.T)

Equipped with this measure function, we are able to register the semantic index to
SuperDuperDB, using already existing models.
Once the semantic index has been created, it may be searched using the ``like`` keyword
in a MongoDB style query.

.. code-block:: python

    >>> from my_package.measures import css
    >>> docs.create_measure('dot', dot)
    >>> docs.create_semantic_index('resnet-index', ['resnet'], measure='dot')
    >>> docs.find_one(like={'_id': ObjectId('6387bc38477124958d0b97d9')}, n=1)
    ObjectId('6387bc38477124958d0b97d9')
    >>> docs.find_one(
    ...     like={'img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=250x361>},
    ...     n=1,
    ... )['_id']
    ObjectId('6387bc38477124958d0b97d9')

It's also possible to train a semantic-index end-2-end using the ``create_semantic_index`` command.
For this it's necessary to define additionally:
- a **splitter**, which divides a document into a query and retrieved item pair
- a **loss** function, which measures the quality of retrievals quantitatively, and is used for backpropagation.
- optionally **metrics**, which measure the quality of retrieval in an interpretable way.

.. code-block:: python

    def split_images(r):
        index = random.randrange(len(r['images']))
        return {'images': [r['images'][index]]}, {'images': [*r['images'][:index], *r['images'][index:]]}

    def ranking_loss(x, y):
        x = x.div(x.norm(dim=1)[:, None])
        y = y.div(y.norm(dim=1)[:, None])
        similarities = x.matmul(y.T)
        return -torch.nn.functional.log_softmax(similarities, dim=1).diag().mean()

    def r_at_1(x, y):
        return y == x[0]


.. code-block:: python

    >>> from my_package.utils import r_at_1, ranking_loss, split_images
    >>> docs.create_metric('r@1', r_at_1)
    >>> docs.create_loss('ranking', ranking_loss)
    >>> docs.create_semantic_index('my_index', ['resnet'], measure='dot', loss='ranking', metrics=['r@1'])

SuperDuperDB is the most natural environment to implement and deploy semantic search and
models such as
`Retro <https://www.deepmind.com/publications/improving-language-models-by-retrieving-from-trillions-of-tokens>`_
which leverage semantic search.

A *semantic index* is the framework within which SuperDuperDB may be used to perform semantic search.
It consists of one or more models which all output vectorial features and whose outputs can be
meaningfully compared to one another using a *measure*. At least one of these models needs to
be set to ``active=True``, so that we have vectors which form the index over which to search.
Once this is in place, any of the specified models may be used to search over the vector index.

Defining your own semantic index
--------------------------------

Given a collection with models available, it's possible to define a semantic index with already
existing models as follows:

.. code-block:: python

    >>> from my_package.measures import css
    >>> docs.create_measure('css', css)
    >>> docs.create_semantic_index(
    ...    'my_semantic_index',
    ...     models=['my_model_1', 'my_model_2', ...],
    ...     measure='css',
    ... )

Using a semantic index
----------------------

To set a default index, update ``docs['_meta']``:

.. code-block:: python

    >>> docs['_meta'].update_one({'key': 'semantic_index'},
    ...                          {'$set': {'value': 'my_semantic_index'}},
    ...                          upsert=True)

To set an index for a query, set the property ``docs.semantic_index = 'my_semantic_index'``.
To use a semantic index, use the ``like`` keyword in ``Collection.find`` or ``Collection.find_one``:

.. code-block:: python

    >>> docs.find(exact_filter, like=doc_contents, n=n)

The ``like`` keyword is passed to one of the models registered during the ``create_semantic_index`` call,
and encoded as a vector. This vector is compared using the ``measure`` argument with the
vectors which have pre-computed using the ``active=True`` model in the *semantic index*.
Which of the models is used to encode the ``document`` is determined by the ``key`` argument of
the ``create_model`` call. SuperDuperDB takes the first model in the *semantic_index* whose ``key``
is in the ``document`` subfield of the query. The ``exact_part`` of the query is executed as a
standard MongoDB query, and the results are merged with the results of the ``$like`` part.

This functionality allows for very sophisticated document filtering using a combination of logic
and AI.

Creating a neighbourhood
------------------------

Once we have a *semantic index* activated for a collection, it's possible to cache
nearest neighbours in the collection documents, and keep this cache in some sense up-to-date
when new data arrives. The way to do this is by using ``docs.create_neighbourhood``:

.. code-block:: python

    >>> docs.create_neighbourhood('my_neighhours', semantic_index='my_semantic_index', n=10)
    # wait a bit...
    >>> docs.find_one()
    {'_id': ObjectId('6387bc38477124958d0b97d9'),
     ...
     '_like': {'clip': [ObjectId('6387bc38477124958d0b97d9'),
       ObjectId('6387bc38477124958d0b9a98'),
       ObjectId('6387bc38477124958d0be495'),
       ObjectId('6387bc38477124958d0b9f51'),
       ObjectId('6387bc38477124958d0bacc0'),
       ObjectId('6387bc38477124958d0b9982'),
       ObjectId('6387bc38477124958d0ba088'),
       ObjectId('6387bc38477124958d0bbad2'),
       ObjectId('6387bc38477124958d0b9ac1'),
       ObjectId('6387bc38477124958d0b9b3a')]}}

You can see that the neighbours according to ``my_semantic_index`` have been cached in the ``_like``
field of the documents. This can come in very useful, when nearest neighbours are required with
very low latency.
