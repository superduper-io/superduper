import dataclasses as dc
import typing as t
from functools import cached_property

from superduperdb.container.component import Component
from superduperdb.container.encoder import Encoder


@dc.dataclass
class Schema(Component):
    identifier: str
    fields: t.Mapping[str, t.Union[Encoder, str]]

    type_id: t.ClassVar[str] = 'schema'

    @cached_property
    def encoded_types(self):
        return [k for k, v in self.fields.items() if isinstance(v, Encoder)]

    @cached_property
    def trivial(self):
        return not any([isinstance(v, Encoder) for v in self.fields.values()])

    @property
    def encoders(self):
        for v in self.fields.values():
            if isinstance(v, Encoder):
                yield v

    def decode(self, data: t.Mapping[str, t.Any]) -> t.Mapping[str, t.Any]:
        if self.trivial:
            return data
        decoded = {}
        for k, v in data.items():
            if k in self.encoded_types:
                field = self.fields[k]
                assert isinstance(field, Encoder)
                v = field.decode(v)
            decoded[k] = v
        return decoded

    def encode(self, data: t.Mapping[str, t.Any]):
        if self.trivial:
            return data
        encoded_data = {}
        for k, v in data.items():
            if k in self.fields and isinstance(self.fields[k], Encoder):
                field_encoder = self.fields[k]
                assert callable(field_encoder)
                encoded_data.update({k: field_encoder(v).encode()})
            else:
                encoded_data.update({k: v})
        return encoded_data
