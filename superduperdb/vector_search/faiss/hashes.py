import numpy
import os
import torch
import faiss

from superduperdb.vector_search.base import BaseHashSet


os.environ['KMP_DUPLICATE_LIB_OK']='True'


class FaissHashSet(BaseHashSet):
    """
    Faiss hash-set for looking up with vector similarity.

    https://github.com/facebookresearch/faiss

    :param h: ``torch.Tensor``
    :param index: list of IDs
    :param measure: measure to assess similarity {'l2', 'dot', 'css'}
    :param faiss_index: Faiss index object if available (prevents need to fit anew)
    """
    def __init__(self, h, index, measure='l2', faiss_index=None):
        super().__init__(h, index)
        if isinstance(h, list):
            if isinstance(h[0], torch.Tensor):
                h = torch.stack(h)
            elif isinstance(h[0], numpy.ndarray):
                h = numpy.stack(h)
            else:
                NotImplementedError('No support for this array/ tensor type')

        if isinstance(h, torch.Tensor):
            h = h.numpy().astype('float32')

        self.h = h
        self.index = index
        if faiss_index is None:
            if measure == 'css':
                h = h / (numpy.linalg.norm(h, axis=1)[:, None])
            if measure == 'l2':
                faiss_index = faiss.index_factory(self.h.shape[1], 'Flat', faiss.METRIC_L2)
            elif measure in {'css', 'dot'}:
                faiss_index = faiss.index_factory(self.h.shape[1], 'Flat', faiss.METRIC_INNER_PRODUCT)
            else:
                raise NotImplementedError(f'"{measure}" not a supported measure for faiss')
            faiss_index.add(h)
        self.faiss_index = faiss_index

    def find_nearest_from_hashes(self, h, n=100):
        if isinstance(h, torch.Tensor):
            h = h.numpy().astype('float32')
        scores, ix = self.faiss_index.search(h, n)
        _ids = [[self.index[i] for i in sub] for sub in ix]
        return _ids, scores.tolist()