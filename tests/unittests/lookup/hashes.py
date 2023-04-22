import os
import torch
import uuid

from superduperdb.vector_search.hashes import FaissHashSet, HashSet


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def test_faiss_hash_set():
    x = torch.randn(1000, 32)
    ids = [uuid.uuid4() for _ in range(x.shape[0])]

    def l2(x, y):
        return -torch.cdist(x, y, 2)

    h1 = FaissHashSet(x, ids, 'l2')
    h2 = HashSet(x, ids, l2)

    y = torch.randn(32)

    res1 = h1.find_nearest_from_hash(y)
    res2 = h2.find_nearest_from_hash(y)

    assert res1['ix'][0] == res2['ix'][0]
    assert res1['_ids'][0] == res2['_ids'][0]
