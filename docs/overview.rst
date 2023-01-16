************************
Overview of SuperDuperDB
************************

SuperDuperDB is designed for productionizing and developing deep learning and AI models
for use-cases where a tight relation between data and models needs to be maintained.
According to a certain opionated point of view this is every use-case...

^^^^^^^^^^^^^^^^^^
e-Commerce example
^^^^^^^^^^^^^^^^^^

Suppose with have a collection of products, where each product is represented by a record containing
some text and an image. Here's what some sample data might look like:

.. code-block:: python

    >>> import json
    >>> with open('data.json') as f:
    ...     data = json.load(f)
    ...
    >>> import requests
    >>> for i in range(len(data)):
    ...     content = requests.get(data[i]['img']).content
    ...     data[i]['img'] = PIL.Image.frombytes('RGB', (250, 361), content)
    ...
    >>> data[0]
    {'brand': 'Even&Fair',
     'title': 'BODYSUIT - Long sleeved top',
     'img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=250x361>}
    >>> data[1]
    {'brand': 'Reebok',
     'title': 'LEGGING - Leggings',
     'img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=250x361>}

In many use-cases we want to apply AI models which understand text and/ or images.
These models may take various forms:

- Vector space embeddings of individual pieces of text or image
- Embeddings of the whole document
- Classifiers of the type of product
- Classifiers of the type of image or text
- Content suggestion for the text based on the image

In principle the possibilities are endless, and multiply the more structure there is in the data.

To do these tasks, it's not just necessary to add text to the database, as might be the case in,
for instance, setting up an ElasticSearch index. Instead, we'd really like our database to
hold **all** the important bits of content, including the images, and in addition any useful predictions
we make based on AI models taking the whole document or part of it as input.

That means that the system must have a way to encode all of these bits of information in the
database, and also include models which are aware of what's going into the database, so that
they can act on the incoming inputs.

That's where SuperDuperDB excels. After setting up SuperDuperDB correctly to work with this data,
and inserting the documents into the database, the stored documents should contain the full
content of each product, including images, and also model outputs stored as tensors.

.. code-block:: python

    >>> documents.insert_many(data)
    >>> documents.find_one()
    {'_id': ObjectId('6387bc38477124958d0b97d9'),
     'brand': 'Even&Fair',
     'title': 'BODYSUIT - Long sleeved top',
     'img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=250x361>,
     '_outputs': {'img': {'resnet': tensor([0.0064,  0.0055, -0.0140,  ...,  0.0120,  0.0084, -0.0253])},
                          'visual_classifier': 'dark-lighting'},
                 {'_base': {'classifier': 'top',
                            'doc_embed': tensor([-0.9041, -0.7223,  0.3630,  ..., -0.3576,  2.1476,  1.4764])}}}

You can see from this example, that SuperDuperDB is able work directly with the Python
objects and extract these after insertion of the data into the database
(including the images, but this works for arbitrary objects). On insertion of the data SuperDuperDB
computes model outputs which go straight into the documents in the database.

You can see in the displayed example that there are three models which have acted on this document:

* ``resnet``
* ``classifier``
* ``doc_embed``

These models have acted respectively on the ``img`` subfield and in the second two cases the whole
document (``_base``). The first model takes in images and emits a ``torch.tensor``, the second
takes in the whole document and emits a human readable ``str``, the third takes in the whole document
and emits a ``torch.tensor``.

This illustrates how SuperDuperDB can be useful for AI systems with diverse and interacting components.
Models can be configured to injest various parts of or the
whole document, and to have outputs of various types. The execution of the model I/O on the documents
is automated in concert with the database updates and insertions which are made. Models
are also allowed to injest the outputs of other models, enabling users to create systems
with models working in collaboration with one another. For instance, in this case, the ``doc_embed``
model has injested the ``resnet`` features in addition to the ``brand`` and ``title`` fields
in order to create a document embedding for the whole.

Certain types of AI model occupy a special place in SuperDuperDB. These are models which
act together to create a so-called "semantic-index". A semantic index is a collection of vectors
which index (a subset of) the documents. These vectors may be compatible with one of more
AI models which embed in the same space. In the above example, the ``resnet`` model may be
considered part of a semantic index, since its outputs are vectors. If we have already registered
such a semantic index with the database, we can use it for querying documents.

In SuperDuperDB such a query might look like this:

.. code-block:: python

    >>> documents.find({
    ...    '$like': {
    ...        'document': {
    ...            'img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=250x361>
    ...        },
    ...        'n': 10
    ...    },
    ...    'brand': 'Adidas Originals'
    ... })
    <superduperdb.cursor.SuperDuperCursor at 0x157f7a9b0>

This query has two parts. The part under the ``$like`` operator refers to the current semantic
index of the collection. This part of the query encodes the contained ``document`` using
the appropriate model in the same vector space as the stored vectors, and looks for similar items.
In the above case, an image in the field ``img`` is passed, and the registered document for the
``img`` key is ``resnet``. Consequently the image is passed to the ``resnet`` for embedding
into the vector space.
The rest of the query can be seen as a standard MongoDB query. The results are combined and
returned in a cursor object in direct analogy to MongoDB.