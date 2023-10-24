import dataclasses as dc

from superduperdb.container.document import Document
from superduperdb.db.base.cursor import SuperDuperCursor


class IbisDocument(Document):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def outputs(self, *args):
        return self.__getitem__('output')


@dc.dataclass
class SuperDuperIbisCursor(SuperDuperCursor):
    def wrap_document(self, r, encoders):
        """
        Wrap a document in a ``Document``.
        """
        return IbisDocument(Document.decode(r, encoders))
