import dataclasses as dc
from functools import cached_property
import typing as t

from superduperdb.container.component import Component
from superduperdb.container.encoder import Encoder
from superduperdb.db.base.db import DB


@dc.dataclass
class Schema(Component):
    identifier: str
    fields: t.Mapping[str, t.Union[str, Encoder]]

    @cached_property
    def trivial(self):
        return any([isinstance(v, Encoder) for v in self.fields.values()])

    def on_create(self, db: DB):
        if self.trivial:
            return
        mapped_names = {
            k: (f'{k}::_encodable={v.identifier}/{v.version}::' if isinstance(v, Encoder) else v)
            for k, v in self.fields.items()
        }
        db.databackend.create_schema(self.identifier, self.fields, mapped_names)

    def decode(self, data: t.Mapping[str, t.Any]) -> t.Mapping[str, t.Any]:
        if self.trivial:
            return data
        return {
            k: (self.fields[k].decode(v) if isinstance(self.fields[k], Encoder) else v)
            for k, v in data.items()
        }

    def encode(self, data):
        if self.trivial:
            return data
        return {
            k: (self.fields[k].encode.artifact(v) if isinstance(self.fields[k], Encoder) else v)
            for k, v in data.items()
        }
