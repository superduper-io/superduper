import os

import numpy

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

    def __init__(self, h, index, measure='l2', faiss_index=None):
        import faiss

        super().__init__(h, index, measure)
        self.h = self.h.astype('float32')
        if faiss_index is None:
            if measure == 'cosine':
                self.h = self.h / (numpy.linalg.norm(self.h, axis=1)[:, None])
            if measure == 'l2':
                faiss_index = faiss.index_factory(
                    self.h.shape[1], 'Flat', faiss.METRIC_L2
                )
            elif measure in {'cosine', 'dot'}:
                faiss_index = faiss.index_factory(
                    self.h.shape[1], 'Flat', faiss.METRIC_INNER_PRODUCT
                )
            else:
                raise NotImplementedError(
                    f'"{measure}" not a supported measure for faiss'
                )
            faiss_index.add(self.h)
        self.faiss_index = faiss_index

    def find_nearest_from_arrays(self, h, n=100):
        import torch

        if isinstance(h, list):
            h = numpy.array(h).astype('float32')
        if isinstance(h, torch.Tensor):
            h = h.numpy().astype('float32')
        scores, ix = self.faiss_index.search(h, n)
        _ids = [[self.index[i] for i in sub] for sub in ix]
        return _ids, scores.tolist()
