import typing as t

from superduperdb.core.encoder import Encoder
from superduperdb.core.documents import Document
from superduperdb.misc.special_dicts import MongoStyleDict


class SuperDuperCursor:
    _results: t.List[float]

    def __init__(
        self,
        cursor,
        id_field: str,
        types: t.Mapping[str, Encoder],
        features: t.Union[t.Mapping[str, str], None] = None,
        scores: t.Optional[t.Dict[str, float]] = None,
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
            # ruff: noqa: E501
            self._results = sorted(
                self._results,
                key=lambda r: -self.scores[str(r[self.id_field])],  # type: ignore[index]
            )
            self.it = 0

    @staticmethod
    def add_features(r, features):
        r = MongoStyleDict(r)
        for k in features:
            r[k] = r['_outputs'][k][features[k]]
        if '_other' in r:
            for k in features:
                if k in r['_other']:
                    r['_other'][k] = r['_outputs'][k][features[k]]
        return r

    @staticmethod
    def wrap_document(r, types):
        return Document(Document.decode(r, types))

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
            r = self.add_features(r, features=self.features)

        return self.wrap_document(r, self.types)
