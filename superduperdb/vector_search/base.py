import numpy
import torch


class BaseHashSet:
    def __init__(self, h, index, *args, **kwargs):
        if isinstance(h, list) and isinstance(h[0], torch.Tensor):
            h = torch.stack(h)
        elif isinstance(h, list) and isinstance(h[0], numpy.ndarray):
            h = numpy.stack(h)
        self.h = h
        self.index = index
        if index is not None:
            self.lookup = dict(zip(index, range(len(index))))

    @property
    def shape(self):  # pragma: no cover
        return self.h.shape

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
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

