from sddb.lookup.measures import measures


class HashSet:
    def __init__(self, h, index):
        self.h = h
        self.index = index
        self.lookup = dict(zip(index, range(len(index))))

    def find_nearest_from_id(self, id_, measure='css', n=100):
        return self.find_nearest_from_ids([id_], measure, n=n)[0]

    def find_nearest_from_ids(self, ids, measure='css', n=100):
        ix = list(map(self.lookup.__getitem__, ids))
        return self.find_nearest_from_hashes(self.h[ix, :], measure, n=n)

    def find_nearest_from_hash(self, h, measure='css', n=100):
        return self.find_nearest_from_hashes(h[None, :], measure, n=n)[0]

    def find_nearest_from_hashes(self, h, measure='css', n=100):
        similarities = measures[measure](h, self.h)
        scores, ix = similarities.topk(min(n, similarities.shape[1]), dim=1)
        ix = ix.tolist()
        scores = scores.tolist()
        ids = [[self.index[i] for i in sub] for sub in ix]
        return [{'scores': s, 'ix': i, 'ids': id_}
                for s, i, id_ in zip(scores, ix, ids)]

    def __getitem__(self, item):
        ix = [self.lookup[i] for i in item]
        return HashSet(self.h[ix], item)


class FaissHashSet(HashSet):
    """
    https://github.com/facebookresearch/faiss/wiki/Getting-started
    """

    def __init__(self, index, h=None, faiss_index=None):
        if faiss_index is None:
            ...