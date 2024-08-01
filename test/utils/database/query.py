import numpy as np


def test_query_select(db):
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
        check_keys(r, ["x", "y", "z", primary_id, '_fold', 'n'])

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
        assert r['n'] > 5
        check_keys(r, ["x", "y", "n"])

    select = table.select("x", "special-y", "special-n").filter(
        table["special-y"] == 1, table['special-n'] > 5
    )
    results = list(select.execute())
    assert len(results) == 3
    assert [6, 8, 10] == [r["special-n"] for r in results]
