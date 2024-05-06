import dataclasses as dc
import typing as t

from ibis.expr.datatypes import DataType, dtype as _dtype

from superduperdb.base.leaf import Leaf


@dc.dataclass(kw_only=True)
class FieldType(Leaf):
    identifier: t.Union[str, DataType]

    def __post_init__(self, db): 
        super().__post_init__(db)
        if isinstance(self.identifier, DataType):
            self.identifier = self.identifier.name


def dtype(x):
    '''
    Ibis dtype to represent basic data types in ibis
    e.g int, str, etc
    '''
    return FieldType(identifier=_dtype(x))
