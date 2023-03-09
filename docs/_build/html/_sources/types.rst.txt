Types in SuperDuperDB
=====================

A **type** is a Python object registered with a SuperDuperDB collection which manages how
model outputs or database content are converted to and from ``bytes`` so that these may be
stored and retrieved from the database. Creating types is a prerequisite to adding models
which have non-Jsonable outputs to a collection, as well as adding content to the database
of a more sophisticated variety, such as images, tensors and so forth.

Here are two examples of types, which can be very handy
for many AI models:

.. code-block:: python

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


    class PILImage:
        types = (PIL.Image.Image,)

        @staticmethod
        def encode(x):
            buffer = io.BytesIO()
            x.save(buffer, format='png')
            return buffer.getvalue()

        @staticmethod
        def decode(bytes_):
            return PIL.Image.open(io.BytesIO(bytes_))


The classes must be pickleable using python ``pickle`` - SuperDuperDB
stores the pickled object in the database.
In the case above, we've used static methods and class variables, because the class isn't
configurable. However, by using an ``__init__`` signature with meaningful arguments,
it's possible to create flexible type classes.
Equipped with this class, we can now register a type with the collection:

.. code-block:: python

    >>> docs = SuperDuperClient(**opts).my_database.documents
    >>> docs.create_type('float_tensor', FloatTensor())
    >>> docs.create_type('image', PILImage())
    >>> docs.list_types()
    ['float_tensor', 'image']
    # retrieve the type object from the database
    >>> docs.types['float_tensor']
     <my_package.FloatTensor at 0x10bbf9270>

Let's test the `"image"` type by adding a `PIL.Image` object to SuperDuperDB:

.. code-block:: python

    >>> import requests, PIL.Image, io
    >>> bytes_ = requests.get('https://www.superduperdb.com/logos/white.png').content
    >>> image = PIL.Image.open(io.BytesIO(bytes_))
    >>> docs.insert_one({'img': image, 'i': 0)
    >>> docs.find_one({'i': 0})
	{'_id': ObjectId('63fca4325d2a192e05fe154a'),
	 'img': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=531x106>,
	 '_fold': 'train'}

A more efficient approach which gives the same result, it to add the type of the data explicitly like this

.. code-block:: python

	>>> docs.insert_one({'img': {'_content': {'bytes': bytes_, 'type': 'image'}}, 'i': 0})
