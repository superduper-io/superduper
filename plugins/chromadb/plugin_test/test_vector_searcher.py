import uuid

import chromadb
import numpy as np
import pytest
from superduper.backends.base.vector_search import VectorItem

from superduper_chromadb import VectorSearcher as ChromaDBVectorSearcher


@pytest.fixture
def index_data():
    h = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    ids = [str(uuid.uuid4()) for _ in range(h.shape[0])]
    yield h, ids


@pytest.fixture
def teardown():
    yield

    client = chromadb.HttpClient(host="localhost", port=9000)

    client.delete_collection("123456")


@pytest.mark.skip(reason="Requires a running ChromaDB instance on localhost:9000")
def test_index(index_data, teardown):

    from superduper import CFG

    CFG.vector_search_engine = "chromadb://localhost:9000"

    vectors, ids = index_data
    index = ChromaDBVectorSearcher(identifier="123456", measure='cosine', dimensions=3)
    index.add(items=[VectorItem(id=id_, vector=hh) for hh, id_ in zip(vectors, ids)])
    y = np.array([0, 0, 1])

    index.find_nearest_from_array(y, 1)
    res, _ = index.find_nearest_from_array(y, 1)
    assert res[0] == ids[0]

    y = np.array([0.66, 0.66, 0.66])

    index.add(items=[VectorItem(id="new_id", vector=y)])

    index.find_nearest_from_array(y, 1)
    res, _ = index.find_nearest_from_array(y, 1)

    assert res[0] == "new_id"
