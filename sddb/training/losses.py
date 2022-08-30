
class Loss:
    ...


class NegativeLoss(Loss):
    """
    Simple negative loss for word embeddings.

    Mikolov et al. https://arxiv.org/pdf/1310.4546.pdf
    Confusing formula with expectation. See here:
    https://lilianweng.github.io/posts/2017-10-15-word-embedding/
    """
    def __call__(self, x, y):
        sim = x.matmul(y.T)
        pos = sim.diag()
        neg = - sim + pos.diag()
        losses = - (pos.sigmoid().log() + neg.sum(1).div(sim.shape[1] - 1))
        return losses.mean()
