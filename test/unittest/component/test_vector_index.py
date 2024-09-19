def test_vector_index_recovery(db):
    from test.utils.usecase.vector_search import build_vector_index

    build_vector_index(db)

    table = db["documents"]
    primary_id = table.primary_id
    vector_index = "vector_index"
    sample_data = list(table.select().execute())[50]

    # Simulate restart
    del db.cluster.vector_search[vector_index]

    db.cluster.vector_search.initialize()

    out = (
        table.like({"x": sample_data["x"]}, vector_index=vector_index, n=10)
        .select()
        .execute()
    )

    ids = [o[primary_id] for o in list(out)]
    assert len(ids) == 10


def test_vector_index_cleanup(db):
    from test.utils.usecase.vector_search import build_vector_index

    build_vector_index(db)
    vector_index = "vector_index"

    uuid = db.show('vector_index', vector_index, -1)['uuid']

    assert vector_index, uuid in db.cluster.vector_search.list()

    db.remove('vector_index', vector_index, force=True)

    assert vector_index, uuid not in db.cluster.vector_search.list()
