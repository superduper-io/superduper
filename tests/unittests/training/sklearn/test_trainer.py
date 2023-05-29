# ruff: noqa: F401, F811

from tests.fixtures.collection import random_arrays, arrays, empty
from sklearn.svm import SVC
from superduperdb.training.sklearn.trainer import SklearnTrainerConfiguration
from superduperdb.models.sklearn.wrapper import Pipeline


def test_classifier(random_arrays):
    model = Pipeline([('svc', SVC(verbose=True))], 'svc')
    cf = SklearnTrainerConfiguration()
    random_arrays.create_model('my_svm', model)
    random_arrays.create_learning_task(
        ['my_svm', '_identity'], ['x', 'y'], configuration=cf, keys_to_watch=['x']
    )
    print(random_arrays.find_one())
