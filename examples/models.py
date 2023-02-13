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


class SimpleTokenizer:
    def __init__(self, tokens, max_length=15):
        self.tokens = tokens
        self._set_tokens = set(self.tokens)
        self.lookup = dict(zip(self.tokens, range(len(self.tokens))))
        self.dictionary = {k: i for i, k in enumerate(tokens)}
        self.max_length = max_length

    def __len__(self):
        return len(self.tokens)

    def preprocess(self, sentence, pad=True):
        sentence = re.sub('[^a-z]]', '', sentence.lower()).strip()
        words = [x for x in sentence.split(' ') if x and x in self.tokens]
        if pad:
            words = words + ['</s>' for _ in range(len(words) - self.max_length)]
        words = words[:self.max_length]
        tokenized = list(map(self.lookup.__getitem__, words))
        tokenized = tokenized + [len(self) + 1 for _ in range(self.max_length - len(words))]
        return torch.tensor(tokenized)


class ConditionalLM(torch.nn.Module):
    def __init__(self, tokenizer, n_hidden=512, max_length=15, n_condition=1024):
        super().__init__()

        self.tokenizer = tokenizer
        self.n_hidden = n_hidden
        self.embedding = torch.nn.Embedding(len(self.tokenizer) + 2, self.n_hidden)
        self.conditioning_linear = torch.nn.Linear(n_condition, self.n_hidden)
        self.rnn = torch.nn.GRU(self.n_hidden, self.n_hidden, batch_first=True)
        self.prediction = torch.nn.Linear(self.n_hidden, len(self.tokenizer) + 2)
        self.max_length = max_length

    def preprocess(self, r):
        out = {}
        if 'caption' in r:
            out['caption'] = [len(self.tokenizer)]  + self.tokenizer.preprocess(r['caption']).tolist()[:-1]
        else:
            out['caption'] = [len(self.tokenizer)]
        out['caption'] = torch.tensor(out['caption'])
        if 'img' in r:
            out['img'] = r['img']
        return out

    def train_forward(self, r):
        input_ = self.embedding(r['caption'])
        img_vectors = self.conditioning_linear(r['img']).unsqueeze(0)
        rnn_outputs = self.rnn(input_, img_vectors)[0]
        return self.prediction(rnn_outputs)

    def forward(self, r):
        assert len(r['caption'][0, :]) == 1
        input_ = self.embedding(r['caption'])
        img_vectors = self.conditioning_linear(r['img'])
        rnn_outputs = self.rnn(input_, img_vectors.unsqueeze(0))[0][:, -1, :]
        predictions = torch.zeros(input_.shape[0], self.max_length).type(torch.long)
        for i in range(self.max_length):
            logits = self.prediction(rnn_outputs)
            best = logits.topk(1, dim=1)[1][:, 0].type(torch.long)
            predictions[:, i] = best
            if i < self.max_length - 1:
                input_ = self.embedding(predictions[:, -1].unsqueeze(1))
                rnn_outputs = self.rnn(input_, rnn_outputs.unsqueeze(0))[0][:, -1, :]
        return predictions

    def postprocess(self, output):
        output = output.tolist()
        try:
            first_end_token = next(x for x in output if x == len(self.tokenizer) + 2)
            output = output[:first_end_token]
        except StopIteration:
            pass
        output = [x for x in output if x < len(self.tokenizer)]
        return ' '.join(list(map(self.tokenizer.tokens.__getitem__, output)))

