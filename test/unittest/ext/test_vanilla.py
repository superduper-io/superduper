from test.db_config import DBConfig

import pytest

from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.document import Document
from superduperdb.components.model import ObjectModel


@pytest.fixture()
def data_in_db(db):
    X = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    db.execute(
        Collection(identifier='documents').insert_many(
            [Document({'X': x, 'y': yy}) for x, yy in zip(X, y)]
        )
    )
    yield db


def test_function_predict_one():
    function = ObjectModel(object=lambda x: x, identifier='test')
    assert function.predict_one(1) == 1


def test_function_predict():
    function = ObjectModel(object=lambda x: x, identifier='test', signature='singleton')
    assert function.predict([1, 1]) == [1, 1]


# TODO: use table to test the sqldb
@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_function_predict_with_document_embedded(data_in_db):
    function = ObjectModel(
        object=lambda x: x,
        identifier='test',
        model_update_kwargs={'document_embedded': False},
    )
    function.predict_in_db(
        X='X', db=data_in_db, select=Collection(identifier='documents').find()
    )
    out = data_in_db.execute(Collection(identifier='_outputs.X.test').find({}))
    assert [o['_outputs']['X']['test']['0'] for o in out] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_function_predict_without_document_embedded(data_in_db):
    function = ObjectModel(object=lambda x: x, identifier='test')
    function.predict_in_db(
        X='X', db=data_in_db, select=Collection(identifier='documents').find()
    )
    out = data_in_db.execute(Collection(identifier='documents').find({}))
    assert [o['_outputs']['X']['test']['0'] for o in out] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_function_predict_with_flatten_outputs(data_in_db):
    function = ObjectModel(
        object=lambda x: [x, x, x] if x > 2 else [x, x],
        identifier='test',
        model_update_kwargs={'document_embedded': False},
        flatten=True,
    )
    function.predict_in_db(
        X='X', db=data_in_db, select=Collection(identifier='documents').find()
    )
    out = data_in_db.execute(Collection(identifier='_outputs.X.test').find({}))
    out = [o for o in out]
    input_ids = [
        c['_id'] for c in data_in_db.execute(Collection(identifier='documents').find())
    ]
    source_ids = []
    for i, id in enumerate(input_ids):
        ix = 3 if i + 1 > 2 else 2
        source_ids.append([id] * ix)
    source_ids = sum(source_ids, [])

    assert [o['_outputs']['X']['test']['0'] for o in out] == [
        1,
        1,
        2,
        2,
        3,
        3,
        3,
        4,
        4,
        4,
        5,
        5,
        5,
    ]
    assert [o['_source'] for o in out] == source_ids


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_function_predict_with_mix_flatten_outputs(data_in_db):
    function = ObjectModel(
        object=lambda x: x if x < 2 else [x, x, x],
        identifier='test',
        flatten=True,
        model_update_kwargs={'document_embedded': False},
    )
    function.predict_in_db(
        X='X', db=data_in_db, select=Collection(identifier='documents').find()
    )
    out = data_in_db.execute(Collection(identifier='_outputs.X.test').find({}))
    out = [o for o in out]
    input_ids = [
        c['_id'] for c in data_in_db.execute(Collection(identifier='documents').find())
    ]
    source_ids = []
    for i, id in enumerate(input_ids):
        source_ids.append(id if i + 1 < 2 else [id] * 3)
    source_ids = source_ids[:1] + sum(source_ids[1:], [])

    assert [o['_outputs']['X']['test']['0'] for o in out] == [
        1,
        2,
        2,
        2,
        3,
        3,
        3,
        4,
        4,
        4,
        5,
        5,
        5,
    ]
    assert [o['_source'] for o in out] == source_ids
