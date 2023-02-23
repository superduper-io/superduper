*********************
SuperDuperDB Concepts
*********************

Content in SuperDuperDB is divided into databases and collections just as in MongoDB.
The key object is ``superduperdb.collection.Collection`` which subclasses ``pymongo.collection.Collection``.
This means that all standard MongoDB collection functionality is also available for a
SuperDuperDB collection. See `here <https://www.mongodb.com/docs/manual/introduction/>`_ for an introduction to MongoDB.

These are the key concepts in SuperDuperDB which are geared towards using and managing AI
models:

* Types
* Models
* Semantic Indexes
* Imputations
* Jobs

Types
=====

A type is an Python object registered with a SuperDuperDB collection which manages how
model outputs or database content are converted to and from ``bytes`` so that these may be
stored and retrieved from the database. Creating types is a prerequisite to adding models
which have non-Jsonable outputs to a collection, as well as adding content to the database
of a more sophisticated variety, such as images, tensors and so forth.

A type is any class with ``.encode`` and ``.decode`` methods
as well as an optional ``.types`` property. Here are two examples of types, which can be very handy
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
stores the pickle object in the database. See here for an summary of what's pickleable
and what is not.
In this case, we've used static methods and class variables, because the class isn't
configurable. However, by using an ``__init__`` signature with meaningful arguments,
it's possible to create flexible type classes.
Equipped with this class, we can now register a type with the collection:

.. code-block:: python

    >>> docs = SuperDuperClient(**opts).my_database.documents
    >>> docs.create_type('float_tensor', FloatTensor())
    >>> docs.create_type('image', PILImage())
    >>> docs.list_types()
    ['float_tensor']
    # retrieve the type object from the database
    >>> docs.types['float_tensor']
     <my_package.FloatTensor at 0x10bbf9270>



Models
======

SuperDuperDB models leverage PyTorch for forward passes, but also can (optionally)
include pre- and post-postprocessing. Here is a CNN classifier example, using the ``torchvision``
library. We define two models, the first a visual embedding model, and the second a classifier
based on the first model's outputs.

.. code-block:: python

    from torchvision import models as visionmodels
    from torchvision import transforms
    from torchvision.transforms.functional import pad
    from torch import nn


    class CNN(nn.Module):
        def __init__(self, width=224, height=224):
            super().__init__()

            resnet = visionmodels.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]
            self.resnet = nn.Sequential(*modules)

            self.normalize_values = \
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.width = width
            self.height = height
            self.to_tensor = transforms.ToTensor()
            self.labels = labels

        def normalize_size(self, image):
            width_ratio = self.width / image.width
            height_ratio = self.height / image.height
            ratio = min(width_ratio, height_ratio)
            image = image.resize((math.floor(ratio * image.width), math.floor(ratio * image.height)))

            p_top = math.floor((self.height - image.height) / 2)
            p_bottom = math.ceil((self.height - image.height) / 2)
            p_left = math.floor((self.width - image.width) / 2)
            p_right = math.ceil((self.width - image.width) / 2)
            image = pad(image,
                        [p_left, p_top, p_right, p_bottom],
                        fill=0,
                        padding_mode='edge')
            return image

        def forward(self, x):
            return self.resnet(x)[:, :, 0, 0]

        def preprocess(self, image):
            image = image.convert("RGB")
            image = self.normalize_size(image)
            image = self.to_tensor(image)
            return self.normalize_values(image)


    class VisualClassifier(torch.nn.Module):
        def __init__(self, labels):
            super().__init__()

            self.linear = torch.nn.Linear(2048, len(labels))
            self.labels = labels

        def preprocess(self, x):
            return x

        def forward(self, x):
            return self.linear(x)

        def postprocess(self, prediction)
            return self.labels[prediction.topk(1)[1].item()]


In order to register these models with SuperDuperDB, we do the following:


.. code-block:: python

    >>> from my_packages.models import CNN
    >>> docs.create_model('resnet', CNN(), filter={'img': {'$exists': 1}}, key='img')
    >>> docs.create_model('visual_classifier': VisualClassifier(my_labels),
    ...                   filter={'img': {'$exists': 1},
    ...                   features={'img': 'resnet'}, key='img')
    # wait a bit...
    >>> docs.find_one()
    {'_id': ObjectId('6387bc38477124958d0b97d9'),
     'img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=250x361>,
     '_outputs': {'img': {'resnet': tensor([0.0064,  0.0055, -0.0140,  ...,  0.0120,  0.0084, -0.0253])},
                          'visual_classifier': 'dark-lighting'}}


The ``create_model`` command saves the ``CNN()`` and ``VisualClassifier`` objects to the MongoDB
filesystem and also applies the model to all of the documents which are selected by the ``filter``
parameter (default ``{}`` - all). The second model depends for its input features on the first
model. This is configured via the ``features={...}`` key-word. The fields in the dictionary
are substituted with the model-outputs defined there.

Semantic Indexes
================

Models and their outputs may be used in concert, to make the content of SuperDuperDB collections
searchable. For example, let's make a semantic index on the basis of the ``img`` field above,
utilizing the same ``resnet`` model used before. We first need to create a measure function
which will be used to compare tensor outputs of the contained models:

.. code-block:: python

    def dot(x, y):
        return x.matmul(y.T)

Equipped with this measure function, we are able to register the semantic index to
SuperDuperDB, using already existing models (models may also be created in-line).
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
- zero or more **metrics**, which measure the quality of retrieval in an interpretable way.

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

Imputations
===========

An imputation is a pair of models, where one model is used to predict the output of the other model.
This subsumes many use cases:

* Classification
* Regression
* Autoregressive modelling (language modelling, time-series modelling, ...)
* Generative adversarial learning
* Image segmentation
* Bounding box regression
* ... (there are many possibilities)

Here is what a basic classification example might look like. First we define the model and
the target:

.. code-block:: python

    import torch

    class MyTarget:
        def __init__(self, labels):
            self.lookup = dict(zip(labels, range(len(labels))))

        def preprocess(self, label):
            return torch.tensor(self.lookup[label])


    class MyModel(torch.nn.Module):
        def __init__(self, labels, dim):
            self.labels = labels
            self.layer = torch.nn.Linear(dim, len(self.labels))

        def forward(self, x):
            return self.layer(x)

        def postprocess(self, output):
            estimate = output.topk(1)[1].item()
            return self.labels[estimate]


    def accuracy(x, y):
        return x == y


Now we can train the model. The models are modified in place, and after training, assuming
the learning problem is feasible, ``my_model`` will be able to estimate missing fields in
the ``y`` key, if the ``x`` field contains a tensor of the right dimensionality.

.. code-block:: python

    >>> from my_package.models import MyTarget, MyModel, accuracy
    >>> docs.create_model('my_target', MyTarget(), active=False, key='y')
    >>> docs.create_model('my_model', MyModel(), key='x')
    >>> docs.create_loss('classification_loss', torch.nn.CrossEntropy())
    >>> docs.create_metric('accuracy', accuracy)
    >>> docs.create_imputation(
    ...     model='my_model',
    ...     target='my_target',
    ...     metrics=['accuracy'],
    ...     loss='classification_loss',
    ... )

Jobs
====

Whenever SuperDuperDB does any of the following:

- Data insertion
- Data updates
- Model creation
- Model training
- Calculations

then the engine is required to perform certain longer running computations.
These computations are wrapped as "jobs" and the jobs are carried out asynchronously on
a pool of parallel workers.

When a command is executed which creates jobs, its output will contain the job ids of the jobs
created. For example inserting data, leads to as many jobs as there are models in the database.
Each of these jobs will compute outputs on those data for a single model. The order of the jobs
is determined by which features are necessary for a given model. Those models with no necessary
input features which result from another model go first.

.. code-block:: python

    >>> job_ids = docs.insert_many(data)[1]
    >>> print(job_ids)
    {'resnet': ['5ebf5272-95ac-11ed-9436-1e00f226d551'],
     'visual_classifier': ['69d283c8-95ac-11ed-9436-1e00f226d551']}

The standard output of these asynchronous jobs is logged to MongoDB. One may watch this
output using, for example, ``docs.watch_job(job_ids['resnet'])``.
