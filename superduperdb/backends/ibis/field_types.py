import dataclasses as dc
import typing as t

from ibis.expr.datatypes import DataType, dtype as _dtype

from superduperdb.base.serializable import Serializable


@dc.dataclass
class FieldType(Serializable):
    """Field type to represent the type of a field in a table.

    This is a wrapper around ibis.expr.datatypes.DataType to make it
    serializable.

    :param identifier: The name of the data type.
    """

    identifier: t.Union[str, DataType]

    def __post_init__(self):
        if isinstance(self.identifier, DataType):
            self.identifier = self.identifier.name


def dtype(x):
    """Ibis dtype to represent basic data types in ibis.

    :param x: The data type
    e.g int, str, etc.
    """
    return FieldType(_dtype(x))
