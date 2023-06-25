import torch


def accuracy(x, y):
    if isinstance(x, torch.Tensor):
        x = x.item()
    if isinstance(y, torch.Tensor):
        y = y.item()
    return x == y


class PatK:
    def __init__(self, k):
        self.k = k

    def __call__(self, x, y):
        return y in x[: self.k]
