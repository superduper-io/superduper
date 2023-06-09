# ruff: noqa: F401, F811
from sklearn.svm import SVC

from superduperdb.core.learning_task import LearningTask
from superduperdb.datalayer.mongodb.query import Select
from superduperdb.models.vanilla.wrapper import FunctionWrapper
from superduperdb.training.sklearn.trainer import SklearnTrainingConfiguration
from superduperdb.models.sklearn.wrapper import Pipeline

from tests.fixtures.collection import random_arrays, arrays, empty


def test_classifier(random_arrays):
    model = Pipeline([('svc', SVC(verbose=True))], 'svc')
    identity = FunctionWrapper(lambda x: x, 'identity')
    cf = SklearnTrainingConfiguration('my-sk-cf')
    random_arrays.database.add(cf)
    random_arrays.database.add(identity, serializer='dill')
    random_arrays.database.add(model)
    random_arrays.database.add(
        LearningTask(
            'my-sk-lt',
            models=['svc', 'identity'],
            keys=['x', 'y'],
            training_configuration='my-sk-cf',
            select=Select('documents'),
            metrics=[],
        )
    )
