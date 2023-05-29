import torch

from superduperdb.core.metric import Metric


def accuracy(x, y):
    if isinstance(x, torch.Tensor):
        x = x.item()
    if isinstance(y, torch.Tensor):
        y = y.item()
    return x == y


class PatK(Metric):
    def __init__(self, k):
        super().__init__(f'p@{k}')
        self.k = k

    def __call__(self, x, y):
        return y in x[: self.k]
