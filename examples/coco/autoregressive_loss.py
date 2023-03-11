import torch


def auto_regressive_loss(x, y):
    # start token = x.shape[2] - 2, stop_token = x.shape[2] - 1 (by convention)
    stop_token = x.shape[2] - 1
    x = x.transpose(2, 1)
    losses = torch.nn.functional.cross_entropy(x, y, reduce=False)
    not_stops = torch.ones_like(losses)
    not_stops[:, 1:] = (y[:, :-1] != stop_token).type(torch.long)
    normalizing_factors = not_stops.sum(axis=1).unsqueeze(1)
    av_loss_per_row = (losses * not_stops).div(normalizing_factors).sum(axis=1)
    return av_loss_per_row.mean()

