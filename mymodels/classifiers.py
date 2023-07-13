import torch


class RandomClassifier:

    def preprocess(self, document):
        return torch.rand(1)[0]

    def forward(self, x):
        return x

    def postprocess(self, x):
        x = bool((x > 0.5).item())
        return {True: 'genuine', False: 'fake'}[x]