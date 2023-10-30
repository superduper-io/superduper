import dataclasses as dc

from superduperdb.base.cursor import SuperDuperCursor
from superduperdb.base.document import Document


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
