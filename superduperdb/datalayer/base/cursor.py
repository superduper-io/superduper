import typing as t

from superduperdb.core import Encoder
from superduperdb.core.documents import Document
from superduperdb.misc.special_dicts import MongoStyleDict


class SuperDuperCursor:
    def __init__(
        self,
        cursor,
        id_field: str,
        types: t.Mapping[str, Encoder],
        features: t.Optional[t.Dict] = None,
        scores: t.Optional[t.List[float]] = None,
    ):
        self.cur = cursor
        self.features = features
        self.scores = scores
        self.types = types
        self.id_field = id_field

        if self.scores is not None:
            self._results = []
            while True:
                try:
                    self._results.append(self.cur.next())
                except StopIteration:
                    break
            self._results = sorted(
                self._results, key=lambda r: -self.scores[str(r[self.id_field])]
            )
            self.it = 0

    def _add_features(self, r):
        r = MongoStyleDict(r)
        for k in self.features:
            r[k] = r['_outputs'][k][self.features[k]]
        if '_other' in r:
            for k in self.features:
                if k in r['_other']:
                    r['_other'][k] = r['_outputs'][k][self.features[k]]
        return r

    def __iter__(self):
        return self

    def __next__(self):
        if self.scores is not None:
            try:
                r = self._results[self.it]
            except IndexError:
                raise StopIteration
            self.it += 1
        else:
            r = self.cur.next()
        if self.scores is not None:
            r['_score'] = self.scores[str(r[self.id_field])]
        if self.features is not None and self.features:
            r = self._add_features(r)
        return Document(Document.decode(r, self.types))
