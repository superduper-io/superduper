import uuid

import pytest
import numpy as np

from superduperdb.vector_search.table_scan import VanillaVectorIndex
from superduperdb.vector_search.table_scan import l2, dot, cosine

@pytest.mark.parametrize("measure", [l2, dot, cosine])
def test_vaniila_index(measure):
    x = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    ids = [uuid.uuid4() for _ in range(x.shape[0])]
    h2 = VanillaVectorIndex(x, ids, measure)
    y = np.array([0, 0.5, 0.5])
    res2, _ = h2.find_nearest_from_array(y, 1)
    assert res2[0] == ids[0]
