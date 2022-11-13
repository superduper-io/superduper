import pickle


def load(r):
    with open(r['path'], 'rb') as f:
        return pickle.load(f)


def save(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
