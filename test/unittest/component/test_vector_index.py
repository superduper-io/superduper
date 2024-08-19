def test_vector_index_recovery(db):
    from test.utils.usecase.vector_search import build_vector_index

    build_vector_index(db)

    table = db["documents"]
    primary_id = table.primary_id
    vector_index = "vector_index"
    sample_data = list(table.select().execute())[50]

    # Simulate restart
    del db.fast_vector_searchers[vector_index]

    db.load('vector_index', vector_index)

    out = (
        table.like({"x": sample_data["x"]}, vector_index=vector_index, n=10)
        .select()
        .execute()
    )

    ids = [o[primary_id] for o in list(out)]
    assert len(ids) == 10
