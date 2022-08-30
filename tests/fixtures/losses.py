import torch


def dummy_loss(x, y):
    return torch.randn(1).item()
