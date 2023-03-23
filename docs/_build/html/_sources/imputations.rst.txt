Imputations for filling in data
=================================

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

        def __call__(self, label):
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
    >>> docs.create_function('my_target', MyTarget())
    >>> docs.create_model('my_model', MyModel())
    >>> docs.create_loss('classification_loss', torch.nn.CrossEntropy())
    >>> docs.create_metric('accuracy', accuracy)
    >>> docs.create_imputation(
    ...     model='my_model',
    ...     model_key='x',
    ...     target='my_target',
    ...     target_key='y',
    ...     metrics=['accuracy'],
    ...     loss='classification_loss',
    ... )

Once the imputation has finished training, by default, SuperDuperDB creates a :ref:`watcher <Watchers in SuperDuperDB>` based on the
``model`` and ``model_key`` parameters.