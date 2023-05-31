import numpy
import torch


class BaseHashSet:
    name = None

    def __init__(self, h, index, measure):
        if isinstance(h, list) and isinstance(h[0], torch.Tensor):
            h = torch.stack(h).numpy()
        elif isinstance(h, list) and isinstance(h[0], numpy.ndarray):
            h = numpy.stack(h)
        elif isinstance(h, list) and isinstance(h[0], list):
            h = numpy.array(h)
        elif isinstance(h, torch.Tensor):
            h = h.numpy()
        self.h = h
        self.index = index
        if index is not None:
            self.lookup = dict(zip(index, range(len(index))))
        self.measure = measure

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
        if isinstance(h, list):
            h = numpy.array(h)
        elif isinstance(h, torch.Tensor):
            h = h.numpy()
        _ids, scores = self.find_nearest_from_hashes(h[None, :], n=n)
        return _ids[0], scores[0]

    def find_nearest_from_hashes(self, h, n=100):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

