from test.utils.usecase.vector_search import add_data

from superduper import model


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


def test_initialize_output_datatype_with_dimensions(db):
    add_data(db, 0, 100)
    db.cfg.auto_schema = True

    import numpy

    @model
    def test(x):
        return numpy.random.randn(32)

    vector_index = test.to_vector_index(key='x', select=db['documents'].select())
    vector_index.pre_create(db)

    from superduper import Table

    assert isinstance(
        vector_index.indexing_listener.output_table,
        Table,
    )

    assert vector_index.dimensions == 32
