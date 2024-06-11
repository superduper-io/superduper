import random
from test.db_config import DBConfig

import numpy
import pytest
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from superduperdb.backends.mongodb.query import MongoQuery
from superduperdb.base.document import Document
from superduperdb.ext.sklearn.model import Estimator, SklearnTrainer


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
        yield Estimator(
            identifier='my-svc',
            object=Pipeline([('my-encoding', Lookup(dictionary)), ('my-svc', SVC())]),
            trainer=SklearnTrainer(
                'my-trainer',
                key=('X', 'y'),
                select=MongoQuery(table='documents').find(),
            ),
        )

    @pytest.fixture()
    def X(self, dictionary):
        yield [random.choice(dictionary) for _ in range(100)]

    @pytest.fixture()
    def data_in_db(self, db, X, y):
        db.execute(
            MongoQuery(table='documents').insert_many(
                [Document({'X': x, 'y': yy}) for x, yy in zip(X, y)]
            )
        )
        yield db

    @pytest.fixture()
    def y(self):
        yield (numpy.random.rand(100) > 0.5).astype(int).tolist()

    # TODO: Test the sqldb
    @pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
    def test_fit_db(self, pipeline, data_in_db):
        _ = data_in_db.apply(pipeline)
