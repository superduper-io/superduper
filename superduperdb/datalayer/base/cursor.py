import dataclasses as dc
import typing as t
from functools import cached_property
from pymongo.cursor import Cursor
from superduperdb.core.documents import Document
from superduperdb.core.encoder import Encoder
from superduperdb.misc.special_dicts import MongoStyleDict


@dc.dataclass
class SuperDuperCursor:
    raw_cursor: Cursor
    id_field: str
    types: t.Dict[str, Encoder] = dc.field(default_factory=dict)
    features: t.Optional[t.Dict[str, str]] = None
    scores: t.Optional[t.List[float]] = None
    _it: int = 0

    @cached_property
    def _results(self) -> t.Optional[t.List[t.Dict]]:
        def key(r):
            # TODO: this is an actual error, because self.scores is a list
            return -self.scores[str(r[self.id_field])]  # type: ignore

        return None if self.scores is None else sorted(self.raw_cursor, key=key)

    def _add_features(self, r):
        r = MongoStyleDict(r)
        for k in self.features:
            r[k] = r['_outputs'][k][self.features[k]]
        if '_other' in r:
            for k in self.features:
                if k in r['_other']:
                    r['_other'][k] = r['_outputs'][k][self.features[k]]
        return r

    def __hash__(self):
        return super().__hash__()

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
            r = self._add_features(r)
        return Document(Document.decode(r, self.types))
