import torch


def ranking_loss(x, y):
    similarities = x.T.matmul(y)
    target = torch.arange(0, x.shape[0]).type(torch.long)
    return torch.nn.functional.cross_entropy(similarities, target)
