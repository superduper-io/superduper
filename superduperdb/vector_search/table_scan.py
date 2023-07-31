import numpy

from superduperdb import logging
from superduperdb.vector_search.base import BaseVectorIndex


class VanillaVectorIndex(BaseVectorIndex):
    """
    Simple hash-set for looking up with vector similarity.

    :param h: ``torch.Tensor``
    :param index: list of IDs
    :param measure: measure to assess similarity
    """

    name = 'vanilla'

    def __init__(self, h, index, measure='cosine'):
        if isinstance(measure, str):
            measure = measures[measure]
        super().__init__(h, index, measure)

    def find_nearest_from_arrays(self, h, n=100):
        similarities = self.measure(h, self.h)  # mypy: ignore
        logging.debug(similarities)
        scores = -numpy.sort(-similarities, axis=1)[:, :n]
        ix = numpy.argsort(-similarities, axis=1)[:, :n]
        ix = ix.tolist()
        scores = scores.tolist()
        _ids = [[self.index[i] for i in sub] for sub in ix]
        return _ids, scores

    def __getitem__(self, item):
        ix = [self.lookup[i] for i in item]
        return VanillaVectorIndex(self.h[ix], item, self.measure)


def l2(x, y):
    return numpy.array([-numpy.linalg.norm(x - y, axis=1)])


def dot(x, y):
    return numpy.dot(x, y.T)


def cosine(x, y):
    x = x / numpy.linalg.norm(x, axis=1)[:, None]
    y = y / numpy.linalg.norm(y, axis=1)[:, None]
    return dot(x, y)


measures = {'cosine': cosine, 'dot': dot, 'l2': l2}
