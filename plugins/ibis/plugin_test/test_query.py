from test.utils.setup.fake_data import add_listeners, add_models, add_random_data

import numpy as np
import pytest
from superduper.base.document import Document
from superduper.components.schema import Schema
from superduper.components.table import Table


def test_serialize_table():
    schema = Schema(
        identifier="my_schema",
        fields={
            "id": "int64",
            "health": "int32",
            "age": "int32",
        },
    )

    s = schema.encode()
    ds = Document.decode(s).unpack()
    assert isinstance(ds, Schema)

    t = Table(identifier="my_table", schema=schema)

    s = t.encode()

    ds = Document.decode(s).unpack()
    assert isinstance(ds, Table)


# TODO: Do we need this test?
@pytest.mark.skip
def test_auto_inference_primary_id():
    s = Table("other", primary_id="other_id")
    t = Table("test", primary_id="id")

    q = t.join(s, t.id == s.other_id)

    assert q.primary_id == ["id", "other_id"]

    q = t.join(s, t.id == s.other_id).group_by("id")

    assert q.primary_id == "other_id"


def test_renamings(db):
    db.cfg.auto_schema = True
    add_random_data(db, n=5)
    add_models(db)
    add_listeners(db)
    t = db["documents"]
    listener_uuid = [db.load('listener', k).outputs for k in db.show("listener")][0]
    q = t.select("id", "x", "y").outputs(listener_uuid)
    data = list(db.execute(q))
    assert isinstance(data[0].unpack()[listener_uuid], np.ndarray)


def test_serialize_query(db):
    from superduper_ibis.query import IbisQuery

    t = IbisQuery(db=db, table="documents", parts=[("select", ("id",), {})])

    q = t.filter(t.id == 1).select(t.id, t.x)

    print(Document.decode(q.encode()).unpack())


def test_add_fold(db):
    add_random_data(db, n=10)
    table = db["documents"]
    select_train = table.select("id", "x", "_fold").add_fold("train")
    result_train = db.execute(select_train)

    select_valid = table.select("id", "x", "_fold").add_fold("valid")
    result_valid = db.execute(select_valid)
    result_train = list(result_train)
    result_valid = list(result_valid)
    assert len(result_train) + len(result_valid) == 10


def test_get_data(db):
    add_random_data(db, n=5)
    db["documents"].limit(2)
    db.metadata.get_component("table", "documents")


def test_insert_select(db):
    add_random_data(db, n=5)
    q = db["documents"].select("id", "x", "y").limit(2)
    r = list(db.execute(q))

    assert len(r) == 2
    assert all(all([k in ["id", "x", "y"] for k in x.unpack().keys()]) for x in r)


def test_filter(db):
    add_random_data(db, n=5)
    t = db["documents"]
    q = t.select("id", "y")
    r = list(db.execute(q))
    ys = [x["y"] for x in r]
    uq = np.unique(ys, return_counts=True)

    q = t.select("id", "y").filter(t.y == uq[0][0])
    r = list(db.execute(q))
    assert len(r) == uq[1][0]


def test_execute_complex_query_sqldb_auto_schema(db):
    import ibis

    db.cfg.auto_schema = True

    table = db["documents"]
    table.insert(
        [Document({"this": f"is a test {i}", "id": str(i)}) for i in range(100)]
    ).execute()

    cur = table.select("this").order_by(ibis.desc("this")).limit(10).execute(db)
    expected = [f"is a test {i}" for i in range(99, 89, -1)]
    cur_this = [r["this"] for r in cur]
    assert sorted(cur_this) == sorted(expected)


def test_select_using_ids(db):
    db.cfg.auto_schema = True

    table = db["documents"]
    table.insert(
        [Document({"this": f"is a test {i}", "id": str(i)}) for i in range(4)]
    ).execute()

    basic_select = db['documents'].select()

    assert len(basic_select.tolist()) == 4
    assert len(basic_select.select_using_ids(['1', '2']).tolist()) == 2


def test_select_using_ids_of_outputs(db):
    from superduper import model

    @model
    def my_func(x):
        return x + ' ' + x

    db.cfg.auto_schema = True

    table = db["documents"]
    table.insert(
        [Document({"this": f"is a test {i}", "id": str(i)}) for i in range(4)]
    ).execute()

    listener = my_func.to_listener(key='this', select=db['documents'].select())
    db.apply(listener)

    q1 = db[listener.outputs].select()
    r1 = q1.tolist()

    assert len(r1) == 4

    ids = [x['id'] for x in r1]

    q2 = q1.select_using_ids(ids[:2])
    r2 = q2.tolist()

    assert len(r2) == 2
