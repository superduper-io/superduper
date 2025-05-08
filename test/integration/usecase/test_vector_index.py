import typing as t

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer

from test.utils.usecase.vector_search import add_data, build_vector_index


def test_vector_index(db: "Datalayer"):
    def check_result(out, sample_data):
        ids = [o[primary_id] for o in list(out)]
        assert len(ids) == 10

        assert sample_data[primary_id] in ids
        assert out[0]['score'] > 0.999999

    build_vector_index(db, n=100)

    vector_index = "vector_index"
    table = db["documents"]
    primary_id = table.primary_id.execute()
    sample_data = table.select().filter(table['x'] == 50).execute()[0]

    # test indexing vector search
    out = (
        table.like({"x": sample_data["x"]}, vector_index=vector_index, n=10)
        .select()
        .execute()
    )

    check_result(out, sample_data)

    # test compatible vector_search
    out = (
        table.like({"y": -sample_data["x"]}, vector_index=vector_index, n=10)
        .select()
        .execute()
    )
    check_result(out, sample_data)

    # test adding new data
    out = table.like({"x": 150}, vector_index=vector_index, n=10).select().execute()

    scores = [r['score'] for r in out]
    assert sum(scores) == 0

    # TODO - this is not triggering the update of the component
    add_data(db, 100, 200)

    assert len(db.cluster.vector_search['VectorIndex', vector_index]) == 200

    out = table.like({"x": 150}, vector_index=vector_index, n=1).select().execute()
    result = out[0]
    assert result['x'] == 150
    assert result['score'] > 0.999999
