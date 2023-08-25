import dataclasses as dc
import typing as t

import ibis

from superduperdb.container.schema import Schema
from superduperdb.container.encoder import Encoder

@dc.dataclass
class IbisSchema(Schema):
    encoded_types: t.List = dc.field(default_factory=list)

    def mutate_column(self, column):
        if column in self.encoded_types:
            return f'{column}::_encodable={self.fields[column].identifier}/{self.fields[column].version}::'
        return column

    def map(self):
        if self.trivial:
            return
        assert not any(['_encodable' in k for k in self.fields.keys()]), 'Reserved substring: _encodable'

        mapped_schema = {}
        for k, v in self.fields.items():
            if isinstance(v, Encoder):
                self.encoded_types.append(k)
                mapped_schema[f'{k}::_encodable={v.identifier}/{v.version}::'] = 'binary'
            else:
                mapped_schema[k] = v
        return ibis.schema(mapped_schema)
