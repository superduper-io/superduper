import dataclasses as dc

import pandas

from superduperdb.base.cursor import SuperDuperCursor
from superduperdb.base.document import Document


class IbisDocument(Document):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def outputs(self, *args):
        return self.__getitem__('output')


@dc.dataclass
class SuperDuperIbisResult(SuperDuperCursor):
    def wrap_document(self, r, encoders):
        """
        Wrap a document in a ``Document``.
        """
        return IbisDocument(Document.decode(r, encoders))

    def as_pandas(self):
        return pandas.DataFrame([Document(r).unpack() for r in self.raw_cursor])

    def __getitem__(self, item):
        return self.raw_cursor[item]

    def __len__(self):
        return len(self.raw_cursor)
