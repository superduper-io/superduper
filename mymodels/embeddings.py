import json
import math
import numpy
import re
import torch


class SimpleGlove:
    def __init__(self, size=50):
        self.embeddings = torch.from_numpy(
            numpy.load(f'data/glove.6B/glove.6B.{size}d.vectors.npy')
        ).type(torch.float)
        with open(f'data/glove.6B/glove.6B.{size}d.index.json') as f:
            self.index = json.load(f)
        self.lookup = dict(zip(self.index, range(len(self.index))))

    def eval(self):
        pass

    def preprocess(self, sentence):
        '''
        >>> model = SimpleGlove()
        >>> model.preprocess('This is[ a test.').shape
        torch.Size([50])
        '''
        cleaned = re.sub('[^a-z0-9 ]', ' ',  sentence.lower())
        cleaned = re.sub('[ ]+', ' ',  cleaned)
        words = cleaned.split()
        words = [x for x in words if x in self.index]
        if not words:
            return torch.ones(50).type(torch.float)
        ix = list(map(self.lookup.__getitem__, words))
        vectors = self.embeddings[ix, :]
        return vectors.sum(0)

    def forward(self, tensor):
        return tensor


class Identity:
    def __init__(self):
        pass

    def eval(self):
        pass

    def preprocess(self, r):
        return r['captions'] + 1

    def forward(self, x):
        return x


class AverageOfGloves:
    def __init__(self):
        self.glove = SimpleGlove()

    def eval(self):
        pass

    def preprocess(self, sentences):
        vectors = torch.stack([
            self.glove.preprocess(sentence)
            for sentence in sentences
        ])
        return vectors.sum(0)

    def forward(self, tensor):
        return tensor


class WordEmbeddings(torch.nn.Module):
    def __init__(self, vocabulary, dimension=64):
        super().__init__()
        self.vocabulary = vocabulary
        self.lookup = dict(zip(vocabulary, range(len(vocabulary))))
        self.embedding = torch.nn.Parameter(
            torch.randn(len(vocabulary), dimension).div(1 / math.sqrt(dimension))
        )

    def preprocess(self, x):
        return self.vocabulary[x]

    def forward(self, x):
        return self.embedding[x, :]
