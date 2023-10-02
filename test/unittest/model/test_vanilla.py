import pytest

from superduperdb.container.document import Document
from superduperdb.container.model import Model
from superduperdb.db.mongodb.query import Collection


@pytest.fixture()
def data_in_db(empty):
    X = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    empty.execute(
        Collection(name='documents').insert_many(
            [Document({'X': x, 'y': yy}) for x, yy in zip(X, y)]
        )
    )
    yield empty


def test_function_predict_one():
    function = Model(object=lambda x: x, identifier='test')
    assert function.predict(1, one=True) == 1


def test_function_predict():
    function = Model(object=lambda x: x, identifier='test')
    assert function.predict([1, 1]) == [1, 1]


def test_function_predict_with_document_embedded(data_in_db):
    function = Model(
        object=lambda x: x,
        identifier='test',
        model_update_kwargs={'document_embedded': False},
    )
    function.predict(X='X', db=data_in_db, select=Collection(name='documents').find())
    out = data_in_db.execute(Collection(name='_outputs.X.test').find({}))
    assert [o['_outputs'] for o in out] == [1, 2, 3, 4, 5]


def test_function_predict_without_document_embedded(data_in_db):
    function = Model(object=lambda x: x, identifier='test')
    function.predict(X='X', db=data_in_db, select=Collection(name='documents').find())
    out = data_in_db.execute(Collection(name='documents').find({}))
    assert [o['_outputs']['X']['test'] for o in out] == [1, 2, 3, 4, 5]
