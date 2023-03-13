import re
import torch


class SimpleTokenizer:
    def __init__(self, tokens, max_length=15):
        self.tokens = tokens
        if '<unk>' not in tokens:
            tokens.append('<unk>')
        self._set_tokens = set(self.tokens)
        self.lookup = dict(zip(self.tokens, range(len(self.tokens))))
        self.dictionary = {k: i for i, k in enumerate(tokens)}
        self.max_length = max_length

    def __len__(self):
        return len(self.tokens)

    def preprocess(self, sentence):
        sentence = re.sub('[^a-z]]', '', sentence.lower()).strip()
        words = [x for x in sentence.split(' ') if x]
        words = [x if x in self.tokens else '<unk>' for x in words]
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
        hidden_states = self.conditioning_linear(r['img']).unsqueeze(0)
        predictions = \
            torch.zeros(r['caption'].shape[0], self.max_length).to(r['caption'].device).type(torch.long)
        predictions[:, 0] = r['caption'][:, 0]
        for i in range(self.max_length - 1):
            rnn_outputs, hidden_states = self.rnn(self.embedding(predictions[:, i]).unsqueeze(1),
                                                  hidden_states)
            logits = self.prediction(rnn_outputs)[:, 0, :]
            predictions[:, i + 1] = logits.topk(1, dim=1)[1][:, 0].type(torch.long)
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

