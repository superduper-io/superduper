import dataclasses as dc
import random

import numpy as np
import pytest

from superduper import Application, Document
from superduper.backends.base.query import Query
from superduper.base.constant import KEY_BLOBS
from superduper.components.listener import Listener
from superduper.components.model import ObjectModel, Trainer


class MyTrainer(Trainer):
    training_done = False

    def fit(self, *args, **kwargs):
        with open('_training_done.txt', 'w'):
            pass


@dc.dataclass
class _Tmp(ObjectModel):
    ...


def test_listener_serializes_properly():
    q = Query(table='test').find({}, {})
    listener = Listener(
        identifier="listener",
        model=ObjectModel("test", object=lambda x: x),
        select=q,
        key="test",
    )
    r = listener.encode()

    # check that the result is JSON-able
    import json

    r.pop(KEY_BLOBS)
    print(json.dumps(r, indent=2))


def test_listener_chaining(db):
    db.cfg.auto_schema = True
    table = db['test']

    def insert_random(start=0):
        data = []
        for i in range(5):
            y = int(random.random() > 0.5)
            data.append(
                Document(
                    {
                        "x": i + start,
                        "y": y,
                    }
                )
            )

        db.execute(table.insert(data))

    # Insert data
    insert_random()

    m1 = ObjectModel("m1", object=lambda x: x + 1)
    m2 = ObjectModel("m2", object=lambda x: x + 2)

    listener1 = Listener(
        model=m1,
        select=table.select(),
        key="x",
        identifier="listener1",
    )

    db.apply(listener1)

    listener2 = Listener(
        model=m2,
        select=listener1.outputs_select,
        key=listener1.outputs,
        identifier='listener2',
    )

    db.apply(listener2)

    def check_listener_output(listener, output_n):
        docs = list(db.execute(listener.outputs_select))
        assert len(docs) == output_n
        assert all([listener.outputs in r for r in docs])

    check_listener_output(listener1, 5)
    check_listener_output(listener2, 5)

    insert_random(start=5)

    check_listener_output(listener1, 10)
    check_listener_output(listener2, 10)


@pytest.mark.parametrize(
    "data",
    [
        1,
        "1",
        {"x": 1},
        [1],
        {
            "x": np.array([1]),
        },
        np.array([[1, 2, 3], [4, 5, 6]]),
    ],
)
@pytest.mark.parametrize("flatten", [False, True])
def test_create_output_dest(db, data, flatten):
    db.cfg.auto_schema = True
    table = db["test"]

    m1 = ObjectModel(
        "m1",
        object=lambda x: data if not flatten else [data] * 10,
    )
    q = table.insert([{"x": 1}])

    db.execute(q)

    listener1 = Listener(
        model=m1,
        select=table.select(),
        key="x",
        identifier="listener1",
        flatten=flatten,
    )

    db.apply(listener1)

    doc = list(db.execute(listener1.outputs_select))[0]
    result = Document(doc.unpack())[listener1.outputs]
    assert isinstance(result, type(data))
    if isinstance(data, np.ndarray):
        assert np.allclose(result, data)
    else:
        assert result == data


@pytest.mark.parametrize(
    "data",
    [
        1,
        "1",
        {"x": 1},
        [1],
        {
            "x": np.array([1]),
        },
        np.array([[1, 2, 3], [4, 5, 6]]),
    ],
)
def test_listener_cleanup(db, data):
    db.cfg.auto_schema = True
    table = db["test"]

    m1 = ObjectModel(
        "m1",
        object=lambda x: data,
    )
    q = table.insert([{"x": 1}])

    db.execute(q)

    listener1 = Listener(
        model=m1,
        select=table.select(),
        key="x",
        identifier="listener1",
    )

    db.add(listener1)
    doc = list(db.execute(listener1.outputs_select))[0]
    result = Document(doc.unpack())[listener1.outputs]
    assert isinstance(result, type(data))
    if isinstance(data, np.ndarray):
        assert np.allclose(result, data)
    else:
        assert result == data

    db.remove('listener', listener1.identifier, force=True)
    assert not db.databackend.check_output_dest(listener1.predict_id)


@pytest.fixture
def cleanup():
    yield
    import os

    try:
        os.remove('_training_done.txt')
    except FileNotFoundError:
        pass


def test_listener_chaining_with_trainer(db, cleanup):
    db.cfg.auto_schema = True
    table = db['test']

    def insert_random(start=0):
        data = []
        for i in range(5):
            y = int(random.random() > 0.5)
            data.append(
                Document(
                    {
                        "x": i + start,
                        "y": y,
                    }
                )
            )

        db.execute(table.insert(data))

    # Insert data
    insert_random()

    features = ObjectModel("features", object=lambda x: x + 1)

    trainable_model = _Tmp(identifier="trainable_model", object=lambda x: x + 2)

    features_listener = Listener(
        model=features,
        select=table.select(),
        key="x",
        identifier="listener1",
    )

    select = db[features_listener.outputs].select()
    trainable_model.trainer = MyTrainer(
        'test', select=select, key=features_listener.outputs
    )

    listener2 = Listener(
        upstream=[features_listener],
        model=trainable_model,
        select=select,
        key=features_listener.outputs,
        identifier='listener2',
    )

    db.apply(listener2)

    import os

    assert os.path.exists('_training_done.txt')


def test_upstream_serializes(db):
    upstream_component = ObjectModel("upstream", object=lambda x: x)

    dependent_listener = Listener(
        identifier="dependent",
        model=upstream_component,
        select=db['other'].select(),
        key='y',
        upstream=[upstream_component],
    )

    db.apply(dependent_listener)

    listener = Listener(
        identifier="test-listener",
        model=ObjectModel("test", object=lambda x: x),
        select=db[dependent_listener.outputs].select(),
        key=dependent_listener.outputs,
        upstream=[upstream_component],
    )

    db.apply(listener)

    assert 'upstream' in db.show('model')

    _ = db.show('listener', listener.identifier, -1)


# TODO: Need to fix this test case
# @pytest.mark.skip("This test is not working")
def test_predict_id_utils(db):
    db.cfg.auto_schema = True
    table = db["test"]

    m1 = ObjectModel(
        "m1",
        object=lambda x: x + 0,
    )
    q = table.insert(
        [
            {"x": 1},
            {"x": 2},
            {"x": 3},
        ]
    )

    db.execute(q)

    listener1 = Listener(
        model=m1,
        select=table.select(),
        key="x",
        identifier="listener1",
    )

    db.apply(listener1)

    outputs = "_outputs__listener1"
    # Listener identifier is set as the table name
    select = db[outputs].select()
    docs = select.tolist()
    # docs = list(db.execute(select))
    assert [doc[listener1.outputs] for doc in docs] == [1, 2, 3]

    # Listener identifier is set as the table name and filter is applied
    table = db[outputs].select()
    select = table.filter(table[outputs] > 1)
    docs = select.tolist()
    assert [doc[listener1.outputs] for doc in docs] == [2, 3]

    # Listener identifier is set as the predict_id in outputs()
    select = db["test"].select().outputs('listener1')
    docs = select.tolist()
    assert [doc[listener1.outputs] for doc in docs] == [1, 2, 3]


def test_complete_uuids(db):
    db.cfg.auto_schema = True

    m1 = ObjectModel(
        "m1",
        object=lambda x: x + 0,
    )

    q = db['test'].insert(
        [
            {"x": 1},
            {"x": 2},
            {"x": 3},
        ]
    )

    db.execute(q)

    l1 = Listener(
        model=m1,
        select=db['test'].select(),
        key="x",
        identifier="l1",
    )

    db.apply(l1)

    q = db['test'].outputs('l1')

    qq = q.complete_uuids(db)

    assert f'"{l1.predict_id}"' in str(qq)

    results = q.tolist()

    assert results[0]['_outputs__l1'] == results[0][l1.outputs]


def test_autofill_data_listener(db):
    db.cfg.auto_schema = True

    m = ObjectModel(
        "m1",
        object=lambda x: x + 2,
    )

    db['test'].insert(
        [
            {"x": 1},
            {"x": 2},
            {"x": 3},
        ]
    ).execute()

    l1 = m.to_listener(select=db['test'].select(), key='x', identifier='l1')
    l2 = m.to_listener(
        select=db['_outputs__l1'].select(), key='_outputs__l1', identifier='l2'
    )

    db.apply(l1)
    db.apply(l2)

    assert l2.key == l1.outputs
    assert l1.outputs in str(l2.select)

    app = Application('my-app', components=[l1, l2])

    r = app.encode(metadata=False)

    import pprint

    pprint.pprint({k: v for k, v in r.items() if k not in {'_blobs', '_files'}})

    out = Document.decode(r, db=db).unpack()

    assert isinstance(out, Application)

    assert isinstance(out.components[0], Listener)
    assert isinstance(out.components[1], Listener)

    assert isinstance(out.components[0].model, ObjectModel)
