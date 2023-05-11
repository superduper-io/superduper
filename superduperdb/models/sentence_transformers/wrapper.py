class SentenceTransformerWrapper:
    def __init__(self, pl):
        self.pl = pl

    def predict_one(self, sentence):
        return self.pl.encode(sentence)

    def predict(self, sentences):
        return self.pl.encode(sentences)