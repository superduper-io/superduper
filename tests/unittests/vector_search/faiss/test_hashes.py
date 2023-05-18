import os
from scipy.spatial.distance import cdist
import torch
import uuid

from superduperdb.vector_search.vanilla.hashes import VanillaHashSet
from superduperdb.vector_search.faiss.hashes import FaissHashSet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def test_faiss_hash_set():

    x = torch.randn(1000, 32)
    ids = [uuid.uuid4() for _ in range(x.shape[0])]

    def l2(x, y):
        return -cdist(x, y)

    h1 = FaissHashSet(x, ids, 'l2')
    h2 = VanillaHashSet(x, ids, l2)

    y = torch.randn(32)

    res1, _ = h1.find_nearest_from_hash(y)
    res2, _ = h2.find_nearest_from_hash(y)

    assert res1[0] == res2[0]
