import warnings
from test.utils.usecase.vector_search import build_vector_index

import pytest
from superduper import CFG, superduper
from superduper.base.datatype import Vector

from superduper_snowflake.vector_search import SnowflakeVectorSearcher

try:
    session = SnowflakeVectorSearcher.create_session(CFG.data_backend)
    DO_SKIP = False
except Exception as e:
    warnings.warn(
        f'Could not connect to Snowflake: {e} on {CFG.data_backend}; '
        'skipping snowflake tests.'
    )
    DO_SKIP = True


@pytest.mark.skipif(DO_SKIP, reason='Only snowflake deployments relevant.')
def test_basic_snowflake_search():
    CFG.vector_search_engine = 'snowflake'
    CFG.force_apply = True

    db = superduper()

    d1 = Vector(shape=[300])
    build_vector_index(db, n=10, list_embeddings=True, vector_datatype=d1, measure='l2')

    vector_index = "vector_index"
    table = db["documents"]
    primary_id = table.primary_id
    sample_data = next(table.select().filter(table['x'] == 5).execute())

    # test indexing vector search
    out = (
        table.like({"x": sample_data["x"]}, vector_index=vector_index, n=3)
        .select()
        .execute()
    )

    def check_result(out, sample_data):
        ids = [o[primary_id] for o in list(out)]
        assert len(ids) == 3
        assert sample_data[primary_id] in ids

    check_result(out, sample_data)
