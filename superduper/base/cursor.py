import dataclasses as dc
import typing as t

from superduper import logging
from superduper.base.document import Document

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.components.schema import Schema


@dc.dataclass
class SuperDuperCursor:
    """A wrapper around a raw cursor that adds some extra functionality.

    A cursor that wraps a cursor and returns ``Document`` wrapping
    a dict including ``Encodable`` objects.

    :param raw_cursor: the cursor to wrap
    :param id_field: the field to use as the document id
    :param db: the datalayer to use to decode the documents
    :param scores: a dict of scores to add to the documents
    :param schema: the schema to use to decode the documents
    :param _it: an iterator to keep track of the current position in the cursor,
            Default is 0.
    :param process_func: a function to process the raw cursor output before
    """

    raw_cursor: t.Any
    id_field: str
    db: t.Optional['Datalayer'] = None
    scores: t.Optional[t.Dict[str, float]] = None
    schema: t.Optional['Schema'] = None
    process_func: t.Optional[t.Callable] = None

    _it: int = 0

    def limit(self, *args, **kwargs) -> 'SuperDuperCursor':
        """Limit the number of results returned by the cursor.

        :param args: Positional arguments to pass to the cursor's limit method.
        :param kwargs: Keyword arguments to pass to the cursor's limit method.
        """
        return SuperDuperCursor(
            raw_cursor=self.raw_cursor.limit(*args, **kwargs),
            id_field=self.id_field,
            db=self.db,
            scores=self.scores,
            schema=self.schema,
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
        """Get the next document from the cursor."""
        r = self.cursor_next()
        if self.process_func is not None:
            r = self.process_func(r)
        if self.scores is not None:
            try:
                r['score'] = self.scores[str(r[self.id_field])]
            except KeyError:
                logging.debug(f"No document id found for {r}")

        return Document.decode(r, db=self.db, schema=self.schema)

    def tolist(self):
        """Return the cursor as a list."""
        return list(self)

    next = __next__
