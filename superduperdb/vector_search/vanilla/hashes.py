import numpy

from superduperdb.vector_search.base import BaseHashSet
from superduperdb.misc.logger import logging
from . import measures


class VanillaHashSet(BaseHashSet):
    """
    Simple hash-set for looking up with vector similarity.

    :param h: ``torch.Tensor``
    :param index: list of IDs
    :param measure: measure to assess similarity
    """

    name = 'vanilla'

    def __init__(self, h, index, measure='css'):
        if isinstance(measure, str):
            measure = getattr(measures, measure)
        super().__init__(h, index, measure)

    def find_nearest_from_hashes(self, h, n=100):
        similarities = self.measure(h, self.h)
        logging.debug(similarities)
        scores = -numpy.sort(-similarities, axis=1)[:, :n]
        ix = numpy.argsort(-similarities, axis=1)[:, :n]
        ix = ix.tolist()
        scores = scores.tolist()
        _ids = [[self.index[i] for i in sub] for sub in ix]
        return _ids, scores

    def __getitem__(self, item):
        ix = [self.lookup[i] for i in item]
        return VanillaHashSet(self.h[ix], item, self.measure)
