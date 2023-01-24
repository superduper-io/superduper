import re

from clip import load as load_clip, tokenize as clip_tokenize
import torch


class CLIP(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.model, self.image_preprocess = load_clip(name)

    def preprocess(self, r):
        if isinstance(r, str):
            return clip_tokenize(r, truncate=True)[0, :]
        elif isinstance(r, list) and isinstance(r[0], str):
            return clip_tokenize(' '.join(r), truncate=True)[0, :]
        return self.image_preprocess(r)

    def forward(self, r):
        if len(r.shape) == 2:
            return self.model.encode_text(r)
        return self.model.encode_image(r)


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
