# ruff: noqa: F401, F811
import random

import numpy
from sklearn.svm import SVC
from sklearn.base import TransformerMixin

from superduperdb.core.documents import Document
from superduperdb.datalayer.mongodb.query import Insert, Select
from superduperdb.models.sklearn.wrapper import Pipeline


import pytest
from tests.fixtures.collection import random_arrays, arrays, empty


class Lookup(TransformerMixin):
    def __init__(self, dictionary):
        table = numpy.random.randn(len(dictionary), 32)
        self.dictionary = {d: table[i] for i, d in enumerate(dictionary)}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, list):
            return [self.dictionary[x] for x in X]
        else:
            return self.dictionary[X]


class TestPipeline:

    @pytest.fixture()
    def dictionary(self):
        yield ['apple', 'orange', 'banana', 'toast']

    @pytest.fixture()
    def pipeline(self, dictionary):
        yield Pipeline(
            'my-svc',
            [
                ('my-encoding', Lookup(dictionary)),
                ('my-svc', SVC())
            ]
        )

    @pytest.fixture()
    def X(self, dictionary):
        yield [random.choice(dictionary) for _ in range(100)]

    @pytest.fixture()
    def data_in_db(self, empty, X, y):
        empty.execute(Insert(
            collection='documents',
            documents=[Document({'X': x, 'y': yy}) for x, yy in zip(X, y)]
        ))
        yield empty

    @pytest.fixture()
    def y(self):
        yield (numpy.random.rand(100) > 0.5).astype(int).tolist()

    def test_fit_predict_classic(self, pipeline, X, y):
        pipeline.fit(X, y)
        output = pipeline.predict(X)
        print(output)

    def test_fit_db(self, pipeline, data_in_db):
        pipeline.fit('X', 'y', database=data_in_db, select=Select('documents'))

