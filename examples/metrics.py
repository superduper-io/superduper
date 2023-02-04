

def accuracy(x, y):
    return x == y


class PatK:
    def __init__(self, k):
        self.k = k

    def __call__(self, x, y):
        return y in x[:self.k]


def jacquard_index(x, y):
    return len(set(x).intersection(set(y))) / len(set(x).union(set(y)))

