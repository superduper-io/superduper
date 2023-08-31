import dataclasses as dc

import ibis

from superduperdb.container.encoder import Encoder
from superduperdb.container.schema import Schema


@dc.dataclass
class IbisSchema(Schema):
    def mutate_column(self, column):
        if column in self.encoded_types:
            name = f'{self.fields[column].identifier}/{self.fields[column].version}'
            return f'{column}::_encodable={name}::'
        return column

    def map(self):
        if self.trivial:
            return
        assert not any(
            ['_encodable' in k for k in self.fields.keys()]
        ), 'Reserved substring: _encodable'

        mapped_schema = {}
        for k, v in self.fields.items():
            if isinstance(v, Encoder):
                mapped_schema[
                    f'{k}::_encodable={v.identifier}/{v.version}::'
                ] = 'binary'
            else:
                mapped_schema[k] = v.type
        return ibis.schema(mapped_schema)
