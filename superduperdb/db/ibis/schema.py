import dataclasses as dc

import ibis

from superduperdb.container.schema import Schema
from superduperdb.container.encoder import Encoder

class IbisSchema(Schema):
    type_id: str = 'ibis.schema'
    encoded_types = []

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
