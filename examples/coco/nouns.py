import spacy


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