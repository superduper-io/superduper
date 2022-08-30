import random

import torch


class Dummy(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(10, 10))

    def preprocess(self, r):
        return torch.randn(10)

    def forward(self, x):
        return x.matmul(self.W)


class DummyLabel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 2)
        self.labels = ['apple', 'pear']

    def preprocess(self, r):
        return torch.LongTensor(1).random_(0, 2).item()

    def forward(self, x):
        return x

    def postprocess(self, x):
        return random.choice(self.labels)


class DummyClassifier(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 2)
        self.labels = ['apple', 'pear']

    def preprocess(self, r):
        return torch.randn(10)

    def forward(self, x):
        return self.layer.forward(x)

    def postprocess(self, output):
        output = output.topk(1)[1].item()
        return self.labels[output]

