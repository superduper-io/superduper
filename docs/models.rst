Models - an extension of PyTorch models
=======================================

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

    def postprocess(*args, **kwargs):
        ...

This approach has the advantage of the methods being able to share data from the class's
``__init__`` signature.

.. code-block:: python

    >>> from my_package import preprocess, postprocess
    >>> docs.create_preprocessor('my_preprocess', preprocess)
    >>> docs.create_postprocessor('my_postprocess', postprocess)
    >>> docs.create_model('my_model', torch.nn.Linear(n_input, n_output),
    ...                   preprocess='my_preprocess',
    ...                   postprocess='my_postprocess',
    ...                   type='image')

This has the advantage of modularity as the pre- and postprocessing parts can be shared between
models more easily.

The ``type`` key-word is only necessary if the output type of the postprocess is not supported 
by MongoDB. Read more about types here ``Types in SuperDuperDB``.

CNN example
-----------

Here is a CNN classifier example, using the ``torchvision``
library. 

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
    >>> docs.create_model('resnet', CNN())
    >>> docs.create_model('visual_classifier': VisualClassifier(my_labels))

The ``create_model`` command saves the ``CNN()`` and ``VisualClassifier`` objects to the MongoDB
filesystem. The models are now registered and ready to go. To set up the models to respond to 
incoming data, see the :ref:`section on watchers <Watchers in SuperDuperDB>`.