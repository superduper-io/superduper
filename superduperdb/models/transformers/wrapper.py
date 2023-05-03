from superduperdb.models.base import SuperDuperModel


class TransformersWrapper(SuperDuperModel):
    def __init__(self, pl):
        self.pl = pl

    def predict_one(self, r, **kwargs):
        return self.pl(r, **kwargs)

    def predict(self, docs, **kwargs):
        return self.pl(docs, **kwargs)


class TokenizingFunction:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def __call__(self, sentence):
        return self.tokenizer(sentence, batch=False, **self.kwargs)