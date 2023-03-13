import re
import torch


class AverageOfGloves:
    def __init__(self, embeddings, index):
        self.embeddings = embeddings
        self.index = index
        self.lookup = dict(zip(self.index, range(len(self.index))))

    def preprocess(self, sentence):
        if isinstance(sentence, list):
            sentence = ' '.join(sentence)
        cleaned = re.sub('[^a-z0-9 ]', ' ',  sentence.lower())
        cleaned = re.sub('[ ]+', ' ',  cleaned)
        words = cleaned.split()
        words = [x for x in words if x in self.index]
        if not words:
            return torch.ones(50).type(torch.float)
        ix = list(map(self.lookup.__getitem__, words))
        vectors = self.embeddings[ix, :]
        return vectors.mean(0)