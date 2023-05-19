from tests.fixtures.collection import random_arrays, arrays, empty
from sklearn.svm import SVC
from superduperdb.training.sklearn.trainer import SklearnTrainerConfiguration


def test_classifier(random_arrays):
    model = SVC(verbose=True)
    cf = SklearnTrainerConfiguration()
    random_arrays.create_model('my_svm', model)
    random_arrays.create_learning_task(
        ['my_svm', '_identity'], ['x', 'y'], configuration=cf, keys_to_watch=['x']
    )
    print(random_arrays.find_one())
