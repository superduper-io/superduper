import dataclasses as dc
import typing as t

from superduperdb import CFG, logging
from superduperdb.base.document import Document
from superduperdb.misc.files import load_uris

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


@dc.dataclass
class SuperDuperCursor:
    """A wrapper around a raw cursor that adds some extra functionality.

    A cursor that wraps a cursor and returns ``Document`` wrapping
    a dict including ``Encodable`` objects.

    :param raw_cursor: the cursor to wrap
    :param id_field: the field to use as the document id
    :param db: the datalayer to use to decode the documents
    :param scores: a dict of scores to add to the documents
    :param decode_function: a function to use to decode the documents
    :param _it: an iterator to keep track of the current position in the cursor,
            Default is 0.
    """

    raw_cursor: t.Any
    id_field: str
    db: t.Optional['Datalayer'] = None
    scores: t.Optional[t.Dict[str, float]] = None
    decode_function: t.Optional[t.Callable] = None

    _it: int = 0

    def limit(self, *args, **kwargs) -> 'SuperDuperCursor':
        """Limit the number of results returned by the cursor."""
        return SuperDuperCursor(
            raw_cursor=self.raw_cursor.limit(*args, **kwargs),
            id_field=self.id_field,
            db=self.db,
            scores=self.scores,
            decode_function=self.decode_function,
        )

    def cursor_next(self):
        """Get the next document from the cursor."""
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
        if self.decode_function is not None:
            r = self.decode_function(r)
        if self.scores is not None:
            try:
                r['score'] = self.scores[str(r[self.id_field])]
            except KeyError:
                logging.warn(f"No document id found for {r}")
        # TODO handle with lazy loading
        if CFG.hybrid_storage:
            load_uris(r, datatypes=self.db.datatypes)
        return Document.decode(r, self.db)

    next = __next__
