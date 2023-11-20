import dataclasses as dc
import typing as t

from superduperdb import CFG
from superduperdb.base.document import Document
from superduperdb.components.encoder import Encoder
from superduperdb.misc.files import load_uris


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
    encoders: t.Dict[str, Encoder] = dc.field(default_factory=dict)
    scores: t.Optional[t.Dict[str, float]] = None

    _it: int = 0

    def limit(self, *args, **kwargs) -> 'SuperDuperCursor':
        """
        Limit the number of results returned by the cursor.
        """
        return SuperDuperCursor(
            raw_cursor=self.raw_cursor.limit(*args, **kwargs),
            id_field=self.id_field,
            encoders=self.encoders,
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

    def wrap_document(self, r, encoders):
        """
        Wrap a document in a ``Document``.
        """
        return Document(Document.decode(r, encoders))

    def __iter__(self):
        return self

    def __next__(self):
        r = self.cursor_next()
        if self.scores is not None:
            r['score'] = self.scores[str(r[self.id_field])]
        if CFG.hybrid_storage:
            load_uris(r, CFG.downloads_folder, encoders=self.encoders)
        return self.wrap_document(r, self.encoders)

    next = __next__
