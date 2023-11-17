import dataclasses as dc
import typing as t
from functools import cached_property

from superduperdb.backends.ibis.field_types import dtype
from superduperdb.components.component import Component
from superduperdb.components.encoder import Encoder


@dc.dataclass
class Schema(Component):
    identifier: str
    fields: t.Mapping[str, t.Union[Encoder, str]]
    version: t.Optional[int] = None

    type_id: t.ClassVar[str] = 'schema'

    def pre_create(self, db) -> None:
        for v in self.fields.values():
            if isinstance(v, Encoder):
                db.add(v)
        return super().pre_create(db)

    def __post_init__(self):
        assert self.identifier is not None, 'Schema must have an identifier'
        assert self.fields is not None, 'Schema must have fields'
        self.fields['_fold'] = dtype('String')

    @property
    def raw(self):
        return {
            k: (v.identifier if not isinstance(v, Encoder) else 'Bytes')
            for k, v in self.fields.items()
        }

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
        """
        Decode data using the schema's encoders

        :param data: data to decode
        """

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
        """
        Encode data using the schema's encoders

        :param data: data to encode
        """
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
