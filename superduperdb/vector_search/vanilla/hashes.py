import numpy
import typing as t

from torch import Tensor

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

    def __init__(
        self,
        h: t.Union[t.List[Tensor], numpy.ndarray, Tensor],
        index: t.List[str],
        measure: str = 'css',
    ) -> None:
        if isinstance(measure, str):
            measure = getattr(measures, measure)
        super().__init__(h, index, measure)

    def find_nearest_from_hashes(
        self,
        h: t.Union[numpy.ndarray, Tensor],
        n: int = 100,
    ) -> t.Tuple[t.List[t.List], numpy.ndarray]:
        similarities = self.measure(h, self.h)  # type: ignore
        logging.debug(similarities)
        scores = -numpy.sort(-similarities, axis=1)[:, :n]
        ix = numpy.argsort(-similarities, axis=1)[:, :n]
        ix = ix.tolist()
        scores = scores.tolist()
        _ids = [[self.index[i] for i in sub] for sub in ix]
        return _ids, scores

    def __getitem__(self, item: t.Any, /) -> 'VanillaHashSet':
        ix = [self.lookup[i] for i in item]
        return VanillaHashSet(self.h[ix], item, self.measure)  # type:ignore
