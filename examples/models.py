from clip import load as load_clip, tokenize as clip_tokenize
import re
import spacy
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


class NounWords:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def preprocess(self, sentences):
        sentence = ' '.join(sentences)
        nouns = []
        for w in self.nlp(sentence):
            if w.pos_ == 'NOUN':
                nouns.append(str(w).lower())
        nouns = sorted(list(set(nouns)))
        return nouns


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


class ConditionalLM(torch.nn.Module):
    def __init__(self, tokens, n_hidden=512, max_length=15):
        super().__init__()

        if '<s>' not in tokens:
            tokens.append('<s>')
        if '</s>' not in tokens:
            tokens.append('</s>')
        self.tokens = tokens
        self.dictionary = {k: i for i, k in enumerate(tokens)}

        self.n_hidden = n_hidden
        self.embedding = torch.nn.Embedding(len(self.tokens), self.n_hidden)
        self.rnn = torch.nn.GRU(self.n_hidden, self.n_hidden)
        self.prediction = torch.nn.Linear(self.n_hidden, len(self.tokens))
        self.max_length = max_length

    def preprocess(self, r):
        out = {}
        if 'caption' in r:
            out['caption'] = [self.start] + self.tokenize(r['caption'])
        else:
            out['caption'] = [self.start]
        out['caption'] = out['caption'][:self.max_length]
        out['img'] = r['img']
        return out

    def train_forward(self, r):
        input_ = self.embedding(r['caption'])
        rnn_outputs = self.rnn(input_, r['img'])[0]
        return rnn_outputs

    def forward(self, r):
        input_ = self.embedding(r['caption'])
        rnn_outputs = self.rnn(input_, r['img'])[0][:, -1, :]
        predictions = torch.zeros(input_.shape[0], self.max_length)
        for i in range(self.max_length):
            logits = self.prediction(rnn_outputs)
            best = logits.topk(1, dim=1)[1]
            predictions[:, i] = best
        return predictions

    def postprocess(self, output):
        output = output.tolist()
        try:
            first_end_token = next(x for x in output if x == self.end_token)
            output = output[:first_end_token]
        except StopIteration:
            pass
        return ' '.join(list(map(self.tokens.__getitem__, output)))

