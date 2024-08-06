import numpy as np
import pytest

from superduper.base.document import Document


def test_insert(db):
    db.cfg.auto_schema = True

    # Test insert one
    db["documents"].insert([{"this": "is a test"}]).execute()
    result = list(db["documents"].select("this").execute())[0]
    assert result["this"] == "is a test"

    # Test insert multiple
    db["documents"].insert([{"this": "is a test"}, {"this": "is a test"}]).execute()
    results = list(db["documents"].select("this").execute())
    assert len(results) == 3


def test_read(db):
    def check_keys(data, keys):
        for k in keys:
            assert k in data

    db.cfg.auto_schema = True

    datas = []
    for i in range(1, 11):
        datas.append(
            {
                "x": f"x_{i}",
                "y": int(i % 2 == 0),
                "special-y": int(i % 2 == 0),
                "z": np.random.randn(1),
                "_fold": "valid" if i % 2 == 0 else "train",
                "n": i,
                "special-n": i,
            }
        )

    db["documents"].insert(datas).execute()

    # Test base select
    results = list(db["documents"].select().execute())
    assert len(results) == 10

    primary_id = db["documents"].primary_id
    for r in results:
        check_keys(r, ["x", "y", "z", primary_id, "_fold", "n"])

    # Test field select
    results = list(db["documents"].select("x").execute())
    assert len(results) == 10
    for r in results:
        assert len(r) == 1
        assert r["x"] is not None

    # Test filter select
    table = db["documents"]
    primary_id = table.primary_id
    select = table.select("x", "y", "n").filter(table.y == 1, table.n > 5)
    results = list(select.execute())
    assert len(results) == 3
    assert [6, 8, 10] == [r["n"] for r in results]
    for r in results:
        assert r["y"] == 1
        assert r["n"] > 5
        check_keys(r, ["x", "y", "n"])

    select = table.select("x", "special-y", "special-n").filter(
        table["special-y"] == 1, table["special-n"] > 5
    )
    results = list(select.execute())
    assert len(results) == 3
    assert [6, 8, 10] == [r["special-n"] for r in results]


# TODO:Add delete common function
def test_delete(db):
    pass


# TODO Add update common function
def test_update(db):
    pass


def _build_vector_index(db):
    from superduper import ObjectModel, VectorIndex

    db.cfg.auto_schema = True

    datas = []
    for i in range(100):
        datas.append(
            {
                "x": i,
                "label": int(i % 2 == 0),
            }
        )

    db["documents"].insert(datas).execute()

    def predict(x):
        n = 100
        vector = [0] * n
        for offset in range(5):
            if offset + x < n:
                vector[offset + x] = 1

            if x - offset >= 0:
                vector[x - offset] = 1

        return np.array(vector)

    model = ObjectModel(
        identifier="model",
        object=predict,
    )

    listener = model.to_listener(
        key="x", select=db["documents"].select(), uuid="vector"
    )

    vector_index = VectorIndex(
        identifier="vector_index",
        indexing_listener=listener,
    )

    db.apply(vector_index)


def test_like(db):
    _build_vector_index(db)
    table = db["documents"]
    primary_id = table.primary_id
    vector_index = "vector_index"

    sample_data = list(table.select().execute())[50]

    out = (
        table.like({"x": sample_data["x"]}, vector_index=vector_index, n=10)
        .select()
        .execute()
    )

    scores = out.scores

    ids = [o[primary_id] for o in list(out)]
    assert len(ids) == 10

    assert sample_data[primary_id] in ids
    assert scores[str(sample_data[primary_id])] > 0.999999

    # Pre-like
    out = (
        table.like({"x": sample_data["x"]}, vector_index=vector_index, n=4)
        .select()
        .filter(table["label"] == 0)
        .execute()
    )

    scores = out.scores
    results = list(out)

    assert len(results) == 2

    assert [r["x"] for r in results] == [49, 51]

    # Post-like
    out = (
        table.select()
        .filter(table["label"] == 0)
        .like({"x": sample_data["x"]}, vector_index=vector_index, n=4)
        .execute()
    )

    scores = out.scores
    results = list(out)

    assert len(results) == 4

    assert [r["x"] for r in results] == [47, 49, 51, 53]


def test_serialize_with_image():
    import PIL.Image

    from superduper.backends.base.query import Query
    from superduper.ext.pillow import pil_image

    img = PIL.Image.open("test/material/data/test.png")
    img = img.resize((2, 2))

    r = Document({"img": pil_image(img)})

    q = Query(table="test_coll").like(r).find()
    print(q)

    s = q.encode()

    import pprint

    pprint.pprint(s)

    decode_q = Document.decode(s).unpack()

    print(decode_q)


def test_insert_with_auto_schema(db):
    db.cfg.auto_schema = True
    import numpy as np
    import PIL.Image

    data = {
        "img": PIL.Image.open("test/material/data/test.png"),
        "array": np.array([1, 2, 3]),
    }

    table_or_collection = db["documents"]
    datas = [Document(data)]

    table_or_collection.insert(datas).execute()
    # Make sure multiple insert works
    table_or_collection.insert(datas).execute()
    datas_from_db = list(table_or_collection.select().execute())

    for d, d_db in zip(datas, datas_from_db):
        assert d["img"].size == d_db["img"].size
        assert np.all(d["array"] == d_db["array"])


def test_insert_with_diff_schemas(db):
    db.cfg.auto_schema = True
    import numpy as np
    import PIL.Image

    table_or_collection = db["documents"]
    data = {
        "array": np.array([1, 2, 3]),
    }
    datas = [Document(data)]
    table_or_collection.insert(datas).execute()

    datas_from_db = list(table_or_collection.select().execute())

    assert np.all(datas[0]["array"] == datas_from_db[0]["array"])

    data = {
        "img": PIL.Image.open("test/material/data/test.png"),
    }
    datas = [Document(data)]

    # Do not support different schema
    with pytest.raises(Exception):
        table_or_collection.insert(datas).execute()


def test_auto_document_wrapping(db):
    db.cfg.auto_schema = True
    import numpy as np

    table_or_collection = db["my_table"]
    data = {"x": np.zeros((1))}
    datas = [Document(data)]
    table_or_collection.insert(datas).execute()

    def _check(n):
        c = list(table_or_collection.select().execute())
        assert len(c) == n
        return c

    _check(1)

    # Without `Document` dict data
    table_or_collection.insert([data]).execute()
    _check(2)

    # Without `Document` non dict data
    table_or_collection.insert([np.zeros((1))]).execute()
    c = _check(3)

    gt = np.zeros((1))

    # Auto wrapped _base
    assert "x" in c[-1]
    assert c[-1].unpack()["x"] == gt


def test_model(db):
    from test.utils.setup.fake_data import add_models

    add_models(db)
    import torch

    m = db.load("model", "linear_a")

    m.predict(torch.randn(32))

    from superduper.backends.base.query import Model

    t = torch.randn(32)

    m.predict(t)

    q = Model(table="linear_a").predict(t)

    out = db.execute(q).unpack()

    assert isinstance(out, torch.Tensor)


def test_model_query():
    from superduper.backends.base.query import Model

    q = Model(table="my-model").predict("This is a test")

    r = q.dict()
    assert r["query"] == 'my-model.predict("This is a test")'
