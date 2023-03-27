Semantic indexes
================

Flexible search using models with tensor outputs
------------------------------------------------

When you've created one or more models in SuperDuperDB which have tensor outputs,
SuperDuperDB may be used to search through the data, on the basis of input which
may be accepted by those models. To enable this feature, one creates one or more **semantic indexes**.

Defining a semantic index
^^^^^^^^^^^^^^^^^^^^^^^^^

At a minimum, creating a semantic index requires users to supply one of more models,
the keys which those models will act on, and a **measure** function for comparing the
output of those models:

.. code-block:: python

    >>> from my_package.measures import css
    >>> docs.create_measure('css', css)
    >>> docs.create_semantic_index(
    ...    'my_semantic_index',
    ...     models=['my_model_1', 'my_model_2', ...],
    ...     keys=['key_1', 'key_2', ...],
    ...     measure='css',
    ... )

Using a semantic index
^^^^^^^^^^^^^^^^^^^^^^

To use a semantic index, use the ``like`` keyword in ``Collection.find`` or ``Collection.find_one``:

.. code-block:: python

    >>> docs.find(exact_filter, like=doc_contents, n=n, semantic_index='my_semantic_index')

The ``like`` keyword is passed to one of the models registered during the ``create_semantic_index`` call,
and encoded by that model. This output is compared using the ``measure`` argument with outputs
of the first model's outputs on the first of the ``keys`` parameter.
Which of the models is used to encode the ``document`` is determined by the ``keys`` argument of
the ``create_semantic_index`` call. SuperDuperDB takes the first model in the *semantic_index* whose ``key``
is in the ``like`` keyword. The exact filter part of the query is executed as a
standard MongoDB query, and the results are merged with the results of the ``like`` part.

This functionality allows for very sophisticated document filtering using a combination of logic
and AI.

Creating a neighbourhood
^^^^^^^^^^^^^^^^^^^^^^^^

Once we have a semantic index activated for a collection, it's possible to cache
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

Training a semantic index by fine-tuning models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It's also possible to *train* a semantic-index end-2-end using the ``create_semantic_index`` command.
For this it's additionally necessary to define:

* an **objective** function, which measures the quality of retrievals quantitatively, and is used for backpropagation.
* (optionally) a **splitter**, which divides a document into a query and retrieved item pair.
* (optionally) **metrics**, which measure the quality of retrieval in an interpretable way.

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

