import dataclasses as dc
import random
from test.utils.component import utils as component_utils
from typing import Iterator

import numpy
import pytest
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from superduper import superduper
from superduper.base.base import Base
from superduper.base.datalayer import Datalayer

from superduper_sklearn.model import Estimator, SklearnTrainer


@pytest.fixture
def db() -> Iterator[Datalayer]:
    db = superduper("mongomock://test_db")

    yield db
    db.drop(force=True, data=True)


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


class documents(Base):
    X: str
    y: int
    _fold: str = dc.field(
        default_factory=lambda: {True: 'train', False: 'valid'}[random.random() > 0.2]
    )


class TestPipeline:
    @pytest.fixture()
    def dictionary(self):
        yield ["apple", "orange", "banana", "toast"]

    @pytest.fixture()
    def X(self, dictionary):
        yield [random.choice(dictionary) for _ in range(100)]

    @pytest.fixture()
    def data_in_db(self, db, X, y):

        db.create(documents)
        db.insert([documents(X=x, y=yy) for x, yy in zip(X, y)])

        yield db

    @pytest.fixture()
    def y(self):
        yield (numpy.random.rand(100) > 0.5).astype(int).tolist()

    def test_fit_db(self, dictionary, data_in_db):
        pipeline = Estimator(
            identifier="my-svc",
            object=Pipeline([("my-encoding", Lookup(dictionary)), ("my-svc", SVC())]),
            trainer=SklearnTrainer(
                "my-trainer",
                key=("X", "y"),
                select=data_in_db["documents"].select(),
            ),
        )
        data_in_db.apply(pipeline, force=True)

        r = data_in_db['documents'].get()
        reloaded = data_in_db.load('Estimator', identifier="my-svc")
        prediction = reloaded.predict([r['X']])
        assert prediction in {0, 1}


def test_encode_and_decode(db):
    model = Estimator(
        identifier="my-svc",
        object=Pipeline(
            [
                ("my-encoding", Lookup(["apple", "orange", "banana", "toast"])),
                ("my-svc", SVC()),
            ]
        ),
        trainer=SklearnTrainer(
            "my-trainer",
            key=("X", "y"),
            select=db["document"].select(),
            db=db,
        ),
        db=db,
    )
    decode_model = component_utils.test_encode_and_decode(model)

    assert decode_model.trainer.identifier == "my-trainer"
    assert decode_model.trainer.key == ("X", "y")
    assert decode_model.trainer.select.table == "document"


def test_sklearn(db):
    m = Estimator(
        identifier='test',
        object=SVC(),
    )
    assert 'object' in m._fields
    db.apply(m, force=True)
    assert db.show('Estimator') == ['test']

    reloaded = db.load('Estimator', identifier='test')
    reloaded.setup()
    assert isinstance(reloaded.object, SVC)
