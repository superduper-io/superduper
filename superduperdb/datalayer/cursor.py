import dataclasses as dc
import typing as t
from functools import cached_property

from pymongo.cursor import Cursor

from superduperdb.base.document import Document
from superduperdb.base.encoder import Encoder
from superduperdb.misc.special_dicts import MongoStyleDict


@dc.dataclass
class SuperDuperCursor:
    raw_cursor: Cursor
    id_field: str
    encoders: t.Dict[str, Encoder] = dc.field(default_factory=dict)
    features: t.Optional[t.Dict[str, str]] = None
    scores: t.Optional[t.Dict[str, float]] = None
    _it: int = 0

    @cached_property
    def _results(self) -> t.Optional[t.List[t.Dict]]:
        def key(r):
            return -self.scores[str(r[self.id_field])]

        return None if self.scores is None else sorted(self.raw_cursor, key=key)

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

    def count(self):
        return len(list(self.raw_cursor.clone()))

    def limit(self, *args, **kwargs):
        return SuperDuperCursor(
            raw_cursor=self.raw_cursor.limit(*args, **kwargs),
            id_field=self.id_field,
            encoders=self.encoders,
            features=self.features,
            scores=self.scores,
        )

    @staticmethod
    def wrap_document(r, encoders):
        return Document(Document.decode(r, encoders))

    def __iter__(self):
        return self

    def __next__(self):
        if self.scores is not None:
            try:
                r = self._results[self._it]
            except IndexError:
                raise StopIteration
            self._it += 1
        else:
            r = self.raw_cursor.next()
        if self.scores is not None:
            r['_score'] = self.scores[str(r[self.id_field])]
        if self.features is not None and self.features:
            r = self.add_features(r, features=self.features)

        return self.wrap_document(r, self.encoders)

    next = __next__
