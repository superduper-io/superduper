import os
import typing as t
from uuid import UUID

import faiss
import numpy
import torch

from superduperdb.vector_search.base import BaseVectorIndex

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class FaissVectorIndex(BaseVectorIndex):
    """
    Faiss hash-set for looking up with vector similarity.

    https://github.com/facebookresearch/faiss

    :param h: ``numpy.array``, ``torch.Tensor`` or ``list``
    :param index: list of IDs
    :param measure: measure to assess similarity {'l2', 'dot', 'css'}
    :param faiss_index: Faiss index object if available (prevents need to _fit anew)
    """

    name = 'faiss'

    def __init__(
        self,
        h: torch.Tensor,
        index: t.List[UUID],
        measure: str = 'l2',
        faiss_index: None = None,
    ) -> None:
        super().__init__(h, index, measure)
        self.h = self.h.astype('float32')  # type: ignore[union-attr]
        if faiss_index is None:
            if measure == 'cosine':
                self.h = self.h / (numpy.linalg.norm(self.h, axis=1)[:, None])
            if measure == 'l2':
                faiss_index = faiss.index_factory(
                    self.h.shape[1], 'Flat', faiss.METRIC_L2  # type: ignore[union-attr]
                )
            elif measure in {'cosine', 'dot'}:
                faiss_index = faiss.index_factory(
                    self.h.shape[1],  # type: ignore[union-attr]
                    'Flat',
                    faiss.METRIC_INNER_PRODUCT,
                )
            else:
                raise NotImplementedError(
                    f'"{measure}" not a supported measure for faiss'
                )
            faiss_index.add(self.h)
        self.faiss_index = faiss_index

    def find_nearest_from_arrays(
        self, h: numpy.ndarray, n: int = 100
    ) -> t.Tuple[t.List[t.List[UUID]], t.List[t.List[float]]]:
        import torch

        if isinstance(h, list):
            h = numpy.array(h).astype('float32')
        if isinstance(h, torch.Tensor):
            h = h.numpy().astype('float32')
        scores, ix = self.faiss_index.search(h, n)
        _ids = [[self.index[i] for i in sub] for sub in ix]
        return _ids, scores.tolist()
