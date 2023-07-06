import dataclasses as dc
import typing as t
from functools import cached_property
from pymongo.cursor import Cursor
from superduperdb.core.documents import Document
from superduperdb.core.encoder import Encoder
from superduperdb.misc.special_dicts import MongoStyleDict
from superduperdb.misc.uri_cache import Cached


@dc.dataclass
class SuperDuperCursor(Cached):
    raw_cursor: Cursor = None
    id_field: str = ''
    types: t.Dict[str, Encoder] = dc.field(default_factory=dict)
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

    def limit(self, *args, **kwargs):
        return SuperDuperCursor(
            raw_cursor=self.raw_cursor.limit(*args, **kwargs),
            id_field=self.id_field,
            types=self.types,
            features=self.features,
            scores=self.scores,
        )

    @staticmethod
    def wrap_document(r, types):
        return Document(Document.decode(r, types))

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

        return self.wrap_document(r, self.types)


SuperDuperCursor.erase_fields()
