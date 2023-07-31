import os
import uuid

import pytest
import torch
from scipy.spatial.distance import cdist

from superduperdb.vector_search.faiss_index import FaissVectorIndex
from superduperdb.vector_search.table_scan import VanillaVectorIndex

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@pytest.mark.parametrize('metric', ['l2', 'cosine', 'dot'])
def test_faiss_hash_set(metric):
    x = torch.randn(1000, 32)
    ids = [uuid.uuid4() for _ in range(x.shape[0])]

    def l2(x, y):
        return -cdist(x, y)

    h1 = FaissVectorIndex(x, ids, metric)
    h2 = VanillaVectorIndex(x, ids, metric)

    y = torch.randn(32)

    res1, _ = h1.find_nearest_from_array(y)
    res2, _ = h2.find_nearest_from_array(y)

    assert res1[0] == res2[0]
