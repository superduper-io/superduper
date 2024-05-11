import dataclasses as dc

import pandas

from superduperdb.base.cursor import SuperDuperCursor
from superduperdb.base.document import Document


@dc.dataclass
class SuperDuperIbisResult(SuperDuperCursor):
    """SuperDuperIbisResult class for ibis query results.

    SuperDuperIbisResult represents ibis query results with options
    to unroll results as i.e pandas.
    """

    def as_pandas(self):
        """Unroll the result as a pandas DataFrame."""
        return pandas.DataFrame([Document(r).unpack() for r in self.raw_cursor])

    def __getitem__(self, item):
        return self.raw_cursor[item]

    def __len__(self):
        return len(self.raw_cursor)
