import tempfile
import uuid

import numpy as np
import pytest

from superduper import CFG
from superduper.backends.local.vector_search import InMemoryVectorSearcher
from superduper.vector_search.base import VectorItem


@pytest.fixture
def index_data(monkeypatch):
    with tempfile.TemporaryDirectory() as unique_dir:
        monkeypatch.setattr(CFG, "lance_home", str(unique_dir))
        h = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        ids = [str(uuid.uuid4()) for _ in range(h.shape[0])]
        yield h, ids, unique_dir


@pytest.mark.parametrize(
    "vector_index_cls",
    [InMemoryVectorSearcher],
)
@pytest.mark.parametrize("measure", ["l2", "dot", "cosine"])
def test_index(index_data, measure, vector_index_cls):
    vectors, ids, ud = index_data
    h = vector_index_cls(uuid="123456", measure=measure, dimensions=3)
    h.add(items=[VectorItem(id=id_, vector=hh) for hh, id_ in zip(vectors, ids)])
    y = np.array([0, 0, 1])
    res, _ = h.find_nearest_from_array(y, 1)
    assert res[0] == ids[0]

    y = np.array([0.66, 0.66, 0.66])

    h.add([VectorItem(id="new", vector=y)])
    h.post_create()
    res, _ = h.find_nearest_from_array(y, 1)

    assert res[0] == "new"
