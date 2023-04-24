from tests.fixtures.collection import arrays, empty, random_arrays
from sklearn.svm import SVC
from superduperdb.models.sklearn import Pipeline


def test_predict(random_arrays):

    pl = Pipeline(random_arrays,
                  'svm',
                  steps=[('svm', SVC())],
                  verbose=True,
                  postprocessor=lambda x: list([int(y) for y in x]))
    pl.fit('x', 'y', {'_fold': 'train'})
    pl.predict_and_update('x', 'y', {})

    r = random_arrays.find_one()
    _ = r['_outputs']['y']['svm']
