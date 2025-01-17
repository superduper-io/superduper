import pytest

from superduper.backends.base.query import parse_query


def test_insert(db):
    db.cfg.auto_schema = True

    # Test insert one
    db["documents"].insert([{"this": "is a test"}])
    result = db["documents"].select("this").execute()[0]
    assert result["this"] == "is a test"

    # Test insert multiple
    db["documents"].insert([{"this": "is a test"}, {"this": "is a test"}])
    results = db["documents"].select("this").execute()
    assert len(results) == 3


def test_atomic_parse(db):
    q = db['docs']['x'] == 2
    r = q.dict()

    parsed = parse_query(query=r['query'], documents=r['documents'], db=db)

    assert len(parsed) == 3

    q = db['docs']['x'] == 'a'
    r = q.dict()

    parsed = parse_query(query=r['query'], documents=r['documents'], db=db)

    assert len(parsed) == 3


class ToSave:
    def __init__(self, x):
        self.x = x


def test_encode_decode_data(db):
    db.cfg.auto_schema = True
    db['docs'].insert([{'x': ToSave(i)} for i in range(10)])

    results = db['docs'].execute()

    assert isinstance(results[0]['x'], ToSave)


def test_filter_select(db):
    db.cfg.auto_schema = True

    db['documents'].insert([{'x': i} for i in range(10)])

    t = db['documents']

    pid = t.primary_id.execute()

    q = t.filter(t['x'] == 2).select(pid)

    r = q.execute()[0]

    assert set(r.keys()) == {pid}


def test_filter(db):
    db.cfg.auto_schema = True

    db['documents'].insert([{'x': i} for i in range(10)])

    t = db['documents']

    ######## ==

    q = t.filter(t['x'] == 1)

    results = q.execute()

    assert len(results) == 1

    assert results[0]['x'] == 1

    ######## >

    q = t.filter(t['x'] > 5)

    results = q.execute()

    assert all(r['x'] > 5 for r in results)
    assert not any(r['x'] <= 5 for r in results)

    ######## >=

    q = t.filter(t['x'] >= 5)

    results = q.execute()

    assert all(r['x'] >= 5 for r in results)
    assert not any(r['x'] < 5 for r in results)

    ######## <

    q = t.filter(t['x'] < 5)

    results = q.execute()

    assert all(r['x'] < 5 for r in results)
    assert not any(r['x'] >= 5 for r in results)

    ######## <=

    q = t.filter(t['x'] <= 5)

    results = q.execute()

    assert all(r['x'] <= 5 for r in results)
    assert not any(r['x'] > 5 for r in results)

    ######## isin

    q = t.filter(t['x'].isin([1, 3]))

    results = q.execute()

    assert set(r['x'] for r in results) == {1, 3}

    ######## !=

    q = t.filter(t['x'] != 5)

    results = q.execute()

    assert set(r['x'] for r in results) == {0, 1, 2, 3, 4, 6, 7, 8, 9}


def test_select_one_col(db):
    db.cfg.auto_schema = True

    db['documents'].insert([{'x': i} for i in range(10)])

    q = db['documents'].select('x')

    results = q.execute()

    assert set(results[0].keys()) == {'x'}


def test_select_all_cols(db):
    db.cfg.auto_schema = True

    db['documents'].insert([{'x': i} for i in range(10)])

    q = db['documents'].select()

    results = q.execute()

    assert len(results) == 10


def test_select_table(db):
    db.cfg.auto_schema = True

    db['documents'].insert([{'x': i} for i in range(10)])

    results = db['documents'].execute()
    assert len(results) == 10


def test_ids(db):
    db.cfg.auto_schema = True

    db['documents'].insert([{'x': i} for i in range(10)])

    results = db['documents'].ids()

    assert len(results) == 10

    assert all(isinstance(x, str) for x in results)


def test_subset(db):
    db.cfg.auto_schema = True

    db['documents'].insert([{'x': i} for i in range(10)])

    ids = db['documents'].ids()
    results = db['documents'].subset(ids[:5])

    pid = db['documents'].primary_id.execute()

    assert set([r[pid] for r in results]) == set(ids[:5])

    db['_outputs__a__123456789'].insert(
        [{'_outputs__a__123456789': i + 2, '_source': id} for i, id in enumerate(ids)]
    )

    results = db['documents'].outputs('a__123456789').subset(ids[:5])

    assert set([r[pid] for r in results]) == set(ids[:5])

    assert 'x' in results[0]


def test_outputs(db):
    db.cfg.auto_schema = True

    ids = db['documents'].insert([{'x': i} for i in range(10)])
    db['_outputs__a__123456789'].insert(
        [{'_outputs__a__123456789': i + 2, '_source': id} for i, id in enumerate(ids)]
    )
    outputs = db['documents'].outputs('a__123456789').execute()
    print(outputs)

    for r in outputs:
        assert r['x'] + 2 == r['_outputs__a__123']


def test_read(db):
    import numpy as np

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

    db["documents"].insert(datas)

    # Test base select
    results = db["documents"].select().execute()
    assert len(results) == 10

    primary_id = db["documents"].primary_id.execute()

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
    select = table.select("x", "y", "n").filter(table['y'] == 1, table['n'] > 5)
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
    results = select.execute()
    assert len(results) == 3
    assert [6, 8, 10] == [r["special-n"] for r in results]


def test_like(db):
    from test.utils.usecase.vector_search import build_vector_index

    build_vector_index(db)
    table = db["documents"]
    # primary_id = table.primary_id.execute()
    vector_index = "vector_index"

    sample_data = list(table.select().execute())[50]

    out = (
        table.like({"x": sample_data["x"]}, vector_index=vector_index, n=10)
        .select()
        .execute()
    )

    scores = [r['score'] for r in out]

    primary_id = table.primary_id.execute()

    ids = [o[primary_id] for o in list(out)]

    assert len(ids) == 10

    assert ids[0] == sample_data[primary_id]
    assert scores[0] > 0.999999

    # Pre-like
    out = (
        table.like({"x": sample_data["x"]}, vector_index=vector_index, n=4)
        .select()
        .filter(table["label"] == 0)
        .execute()
    )

    assert len(out) == 2

    assert set(r["x"] for r in out) == {49, 51}

    # Post-like
    out = (
        table.select()
        .filter(table["label"] == 0)
        .like({"x": sample_data["x"]}, vector_index=vector_index, n=4)
        .execute()
    )

    scores = [r['score'] for r in out]

    assert len(out) == 4

    assert set(r["x"] for r in out) == {47, 49, 51, 53}


def test_insert_with_auto_schema(db):
    db.cfg.auto_schema = True
    import numpy as np

    data = {
        "array": np.array([1, 2, 3]),
    }

    table_or_collection = db["documents"]

    table_or_collection.insert([data])

    datas_from_db = list(table_or_collection.select().execute())

    for d, d_db in zip([data], datas_from_db):
        assert np.all(d["array"] == d_db["array"])


def test_insert_with_diff_schemas(db):
    db.cfg.auto_schema = True
    import numpy as np
    import pandas as pd

    table_or_collection = db["documents"]
    data = {
        "array": np.array([1, 2, 3]),
    }
    table_or_collection.insert([data])

    datas_from_db = table_or_collection.select().execute()

    assert np.all(data["array"] == datas_from_db[0]["array"])

    data = {
        "df": pd.DataFrame(np.random.randn(10, 10)),
    }

    # Do not support different schema
    with pytest.raises(Exception):
        table_or_collection.insert(data)


def test_parse_outputs_query(db):
    q = parse_query(
        query='_outputs__listener1__9bc4a01366f24603.select()',
        documents=[],
        db=db,
    )

    assert len(q) == 2
