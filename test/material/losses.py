import torch


def ranking_loss(x, y):
    x = x.div(x.norm(dim=1)[:, None])
    y = y.div(y.norm(dim=1)[:, None])
    similarities = x.matmul(y.T)  # causes a segmentation fault for no reason in pytest
    return -torch.nn.functional.log_softmax(similarities, dim=1).diag().mean()
