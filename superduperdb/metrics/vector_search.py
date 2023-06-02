from superduperdb.core.metric import Metric


class PatK(Metric):
    def __init__(self, k):
        super().__init__(f'p@{k}')
        self.k = k

    def __call__(self, x, y):
        return y in x[: self.k]