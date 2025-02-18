from test.utils.setup.fake_data import add_listeners, add_models, add_random_data

import numpy as np
import pytest
from superduper.base.document import Document
from superduper.components.table import Table


def test_serialize_table():
    fields = {
        "id": "int",
        "health": "int",
        "age": "int",
    }

    t = Table(identifier="my_table", fields=fields)

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
    listener_uuid = [db.load('Listener', k).outputs for k in db.show("Listener")][0]
    q = t.select("id", "x", "y").outputs(listener_uuid.split('__', 1)[-1])
    data = q.execute()
    assert isinstance(data[0].unpack()[listener_uuid], np.ndarray)


def test_serialize_query(db):
    t = db['documents']
    q = t.filter(t['id'] == 1).select('id', 'x')

    print(Document.decode(q.encode(), db=db).unpack())


def test_get_data(db):
    add_random_data(db, n=5)
    db["documents"].limit(2)
    db.metadata.get_component("Table", "documents")


def test_insert_select(db):
    add_random_data(db, n=5)
    q = db["documents"].select("id", "x", "y").limit(2)
    r = q.execute()

    assert len(r) == 2
    assert all(all([k in ["id", "x", "y"] for k in x.unpack().keys()]) for x in r)


def test_filter(db):
    add_random_data(db, n=5)
    t = db["documents"]
    q = t.select("id", "y")
    r = q.execute()
    ys = [x["y"] for x in r]
    uq = np.unique(ys, return_counts=True)

    q = t.select("id", "y").filter(t['y'] == uq[0][0])
    r = q.execute()
    assert len(r) == uq[1][0]


def test_select_using_ids(db):
    db.cfg.auto_schema = True

    table = db["documents"]
    table.insert([{"this": f"is a test {i}", "id": str(i)} for i in range(4)])

    basic_select = db['documents'].select()

    assert len(basic_select.execute()) == 4
    assert len(basic_select.subset(['1', '2'])) == 2


def test_select_using_ids_of_outputs(db):
    from superduper import model

    @model
    def my_func(x):
        return x + ' ' + x

    db.cfg.auto_schema = True

    table = db["documents"]
    table.insert([{"this": f"is a test {i}", "id": str(i)} for i in range(4)])

    listener = my_func.to_listener(key='this', select=db['documents'].select())
    db.apply(listener)

    q1 = db[listener.outputs].select()
    r1 = q1.execute()

    assert len(r1) == 4

    ids = [x['id'] for x in r1]

    r2 = q1.subset(ids[:2])

    assert len(r2) == 2
