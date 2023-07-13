

def accuracy(x, y):
    assert len(x) == len(y)
    return sum([xx == yy for xx, yy in zip(x, y)]) / len(x)