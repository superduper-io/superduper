import dataclasses as dc
import typing as t

from superduperdb import CFG
from superduperdb.base.document import Document
from superduperdb.misc.files import load_uris

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


@dc.dataclass
class SuperDuperCursor:
    """
    A cursor that wraps a cursor and returns ``Document`` wrapping
    a dict including ``Encodable`` objects.

    :param raw_cursor: the cursor to wrap
    :param id_field: the field to use as the document id
    :param encoders: a dict of encoders to use to decode the documents
    :param scores: a dict of scores to add to the documents
    """

    raw_cursor: t.Any
    id_field: str
    db: t.Optional['Datalayer'] = None
    scores: t.Optional[t.Dict[str, float]] = None

    _it: int = 0

    def limit(self, *args, **kwargs) -> 'SuperDuperCursor':
        """
        Limit the number of results returned by the cursor.
        """
        return SuperDuperCursor(
            raw_cursor=self.raw_cursor.limit(*args, **kwargs),
            id_field=self.id_field,
            db=self.db,
            scores=self.scores,
        )

    def cursor_next(self):
        if isinstance(self.raw_cursor, list):
            if self._it >= len(self.raw_cursor):
                raise StopIteration
            r = self.raw_cursor[self._it]
            self._it += 1
            return r
        return self.raw_cursor.next()

    def __iter__(self):
        return self

    def __next__(self):
        r = self.cursor_next()
        if self.scores is not None:
            r['score'] = self.scores[str(r[self.id_field])]
        # TODO handle with lazy loading
        if CFG.hybrid_storage:
            load_uris(r, datatypes=self.db.datatypes)
        return Document.decode(r, self.db)

    next = __next__
