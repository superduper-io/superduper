import random

import torch


class BinaryTarget:
    def eval(self):
        pass

    def forward(self, x):
        return x.type(torch.float)


class BinaryClassifier(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)[:, 0]

    def postprocess(self, x):
        return (x > 0.5).type(torch.float).item()


class ModelAttributes(torch.nn.Module):
    def __init__(self, input_dim, d1, d2):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, d1)
        self.linear2 = torch.nn.Linear(input_dim, d2)


class NoForward:
    def preprocess(self, x):
        return x + torch.randn(32)
