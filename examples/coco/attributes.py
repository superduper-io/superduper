import torch


class FewHot:
    def __init__(self, tokens):
        self.tokens = tokens
        self.lookup = dict(zip(tokens, range(len(tokens))))

    def preprocess(self, x):
        x = [y for y in x if y in self.tokens]
        integers = list(map(self.lookup.__getitem__, x))
        empty = torch.zeros(len(self.tokens))
        empty[integers] = 1
        return empty


class TopK:
    def __init__(self, tokens, n=10):
        self.tokens = tokens
        self.n = n

    def __call__(self, x):
        pred = x.topk(self.n)[1].tolist()
        return [self.tokens[i] for i in pred]