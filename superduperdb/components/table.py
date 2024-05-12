import dataclasses as dc
import typing as t

from superduperdb.components.component import Component
from superduperdb.components.schema import Schema


@dc.dataclass(kw_only=True)
class Table(Component):
    type_id: t.ClassVar[str] = 'table'
    schema: Schema
