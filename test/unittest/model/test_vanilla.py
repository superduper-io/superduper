import pytest

from superduperdb.base.document import Document
from superduperdb.components.model import Model
from superduperdb.db.mongodb.query import Collection


@pytest.fixture()
def data_in_db(empty):
    X = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    empty.execute(
        Collection(identifier='documents').insert_many(
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
    function.predict(
        X='X', db=data_in_db, select=Collection(identifier='documents').find()
    )
    out = data_in_db.execute(Collection(identifier='_outputs.X.test').find({}))
    assert [o['_outputs']['X']['test'] for o in out] == [1, 2, 3, 4, 5]


def test_function_predict_without_document_embedded(data_in_db):
    function = Model(object=lambda x: x, identifier='test')
    function.predict(
        X='X', db=data_in_db, select=Collection(identifier='documents').find()
    )
    out = data_in_db.execute(Collection(identifier='documents').find({}))
    assert [o['_outputs']['X']['test'] for o in out] == [1, 2, 3, 4, 5]


def test_function_predict_with_flatten_outputs(data_in_db):
    function = Model(
        object=lambda x: [x, x, x] if x > 2 else [x, x],
        identifier='test',
        model_update_kwargs={'document_embedded': False},
        flatten=True,
    )
    function.predict(
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

    assert [o['_outputs']['X']['test'] for o in out] == [
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


@pytest.mark.skip
def test_function_predict_with_mix_flatten_outputs(data_in_db):
    function = Model(
        object=lambda x: x if x < 2 else [x, x, x],
        identifier='test',
        flatten=True,
        model_update_kwargs={'document_embedded': False},
    )
    function.predict(
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

    assert [o['_outputs']['X']['test'] for o in out] == [
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
