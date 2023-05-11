from superduperdb.vector_search.base import BaseHashSet
from superduperdb.misc.logger import logging


class VanillaHashSet(BaseHashSet):
    """
    Simple hash-set for looking up with vector similarity.

    :param h: ``torch.Tensor``
    :param index: list of IDs
    :param measure: measure to assess similarity
    """
    def __init__(self, h, index, measure):
        super().__init__(h, index)
        self.measure = measure

    def find_nearest_from_id(self, _id, n=100):
        _ids, scores = self.find_nearest_from_ids([_id], n=n)
        return _ids[0], scores[0]

    def find_nearest_from_ids(self, _ids, n=100):
        ix = list(map(self.lookup.__getitem__, _ids))
        return self.find_nearest_from_hashes(self.h[ix, :], n=n)

    def find_nearest_from_hash(self, h, n=100):
        _ids, scores = self.find_nearest_from_hashes(h[None, :], n=n)
        return _ids[0], scores[0]

    def find_nearest_from_hashes(self, h, n=100):
        similarities = self.measure(h, self.h)
        logging.debug(similarities)
        scores, ix = similarities.topk(min(n, similarities.shape[1]), dim=1)
        ix = ix.tolist()
        scores = scores.tolist()
        _ids = [[self.index[i] for i in sub] for sub in ix]
        return _ids, scores

    def __getitem__(self, item):
        ix = [self.lookup[i] for i in item]
        return VanillaHashSet(self.h[ix], item, self.measure)
