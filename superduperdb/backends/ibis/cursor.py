import dataclasses as dc

import pandas

from superduperdb.base.cursor import SuperDuperCursor
from superduperdb.base.document import Document


@dc.dataclass
class SuperDuperIbisResult(SuperDuperCursor):
    def as_pandas(self):
        return pandas.DataFrame([Document(r).unpack() for r in self.raw_cursor])

    def __getitem__(self, item):
        return self.raw_cursor[item]

    def __len__(self):
        return len(self.raw_cursor)
