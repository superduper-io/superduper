import dataclasses as dc
import typing as t

from ibis.expr.datatypes import DataType, dtype as _dtype

from superduperdb.base.serializable import Serializable


@dc.dataclass
class FieldType(Serializable):
    identifier: t.Union[str, DataType]

    def __post_init__(self):
        if isinstance(self.identifier, DataType):
            self.identifier = self.identifier.name


def dtype(x):
    return FieldType(_dtype(x))
