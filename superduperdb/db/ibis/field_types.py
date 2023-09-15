import dataclasses as dc
import typing as t

from ibis.expr.datatypes import DataType, dtype as _dtype


@dc.dataclass
class FieldType:
    type: t.Union[str, DataType]

    def __post_init__(self):
        if isinstance(self.type, DataType):
            self.type = self.type.name


def dtype(x):
    return FieldType(_dtype(x))
