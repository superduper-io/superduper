def jacquard_index(x, y):
    return len(set(x).intersection(set(y))) / len(set(x).union(set(y)))