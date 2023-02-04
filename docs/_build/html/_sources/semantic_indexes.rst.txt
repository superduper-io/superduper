*******************************
Deep dive into semantic indexes
*******************************

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
================================

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

If the models don't exist, and only make sense within the semantic index, then it's possible to
create them in line with:

.. code-block:: python

    >>> docs.create_semantic_index(
    ...    'my_semantic_index',
    ...     models=[kwargs_1, kwargs_2, ...],
    ...     measure='css',
    ... )

The ``kwargs_<i>`` are passed onto ``docs.create_model``.

Using a semantic index
======================

SuperDuperDB comes with a dialect of the MongoDB query language, augmenting the standard ``dict``
based syntax with an additional operator ``$like``. In order to use ``$like`` in a query,
a default *semantic index* for the collection or for the particular query must be set.

To set a default index, update ``docs['_meta']``:

.. code-block:: python

    >>> docs['_meta'].update_one({'key': 'semantic_index'},
    ...                          {'$set': {'value': 'my_semantic_index'}},
    ...                          upsert=True)

To set an index for a query, set the property ``docs.semantic_index = 'my_semantic_index'``.

The syntax for using ``$like`` is:

.. code-block:: python

    >>> docs.find({
    ...     '$like': {
    ...         'document': {
    ...             **doc_contents
    ...         },
    ...         'n': n  # number of similar items to search for
    ...     },
    ...     **exact_part,
    ... })

The SuperDuperDB client separates the ``$like`` part from the ``exact_part``. The ``document``
subfield is passed to one of the models registered during the ``create_semantic_index`` call,
and encoded as a vector. This vector is compared using the ``measure`` argument with the
vectors which have pre-computed using the ``active=True`` model in the *semantic index*.
Which of the models is used to encode the ``document`` is determined by the ``key`` argument of
the ``create_model`` call. SuperDuperDB takes the first model in the *semantic_index* whose ``key``
is in the ``document`` subfield of the query. The ``exact_part`` of the query is executed as a
standard MongoDB query, and the results are merged with the results of the ``$like`` part.

This procedure allows for very sophisticated document filtering using a combination of logic
and AI.

Creating a neighbourhood
========================

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
