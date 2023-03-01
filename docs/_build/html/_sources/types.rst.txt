Types in SuperDuperDB
=====================

A type is a Python object registered with a SuperDuperDB collection which manages how
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


The classes must be defined so that standard ``pickle`` serialization works - SuperDuperDB
stores the pickle object in the database.
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