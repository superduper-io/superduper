# ruff: noqa: F401, F811

import numpy
from sklearn.svm import SVC

from superduperdb.core.watcher import Watcher
from superduperdb.datalayer.mongodb.query import Select
from superduperdb.models.sklearn.wrapper import Pipeline
from tests.fixtures.collection import random_arrays, arrays, empty, int64


def test_pipeline(random_arrays, int64):
    X = numpy.random.randn(100, 32)
    y = (numpy.random.rand(100) > 0.5).astype(int)
    est = Pipeline([('my-svc', SVC())], 'my-svc')
    est.fit(X, y)
    random_arrays.add(est)
    pl = random_arrays.models['my-svc']
    print(pl)
    random_arrays.add(
        Watcher(select=Select('documents'), model='my-svc', key='x')
    )
