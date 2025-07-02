import numpy
import pytest

from superduper.components.table import Table
from superduper import superduper, CFG
from superduper_postgres.vector_search import PGVectorSearcher
from superduper.backends.base.vector_search import measures


@pytest.fixture
def db():
    _db = superduper(cluster_engine='local', vector_search_engine='postgres', initialize_cluster=False)
    yield _db
    _db.drop(True, True)


def test_pgvector(db):

    vector_table = Table(
        'vector_table',
        fields={
            'id': 'str',
            'vector': 'vector[float:3]',
        },
        primary_id='id',
    )

    db.apply(vector_table, force=True)

    db['vector_table'].insert(
        [
            {'vector': numpy.random.randn(3)}
            for _ in range(10)  
        ]
    )
    retrieved_vectors = db['vector_table'].execute()
    assert isinstance(retrieved_vectors[0]['vector'], numpy.ndarray)

    vector_searcher = PGVectorSearcher(
        table='vector_table',
        vector_column='vector',
        primary_id='id',
        dimensions=3,
        measure='cosine',
        uri=CFG.data_backend
    )

    vector_searcher.initialize()

    h = numpy.random.randn(3)

    result = vector_searcher.find_nearest_from_array(
        h=h,
        n=10
    )

    import time
    time.sleep(2)

    scores_manual = {}
    for r in retrieved_vectors:
        scores_manual[r['id']] = measures['cosine'](r['vector'][None, :], h[None, :]).item()

    best_id = max(scores_manual, key=scores_manual.get)

    vector_searcher.drop()

    assert best_id == result[0][0]
