import dataclasses as dc
from functools import cached_property
import typing as t

from superduperdb.container.component import Component
from superduperdb.container.encoder import Encoder


"""
Table with single image column:

```python
from superduperdb.ext.pillow.image import pil_image

schema = Schema('image_schema', {'img': pil_image})
t = Table(identifier='my_table', schema=schema)
t.create(db)
```

"""


@dc.dataclass
class Schema(Component):
    identifier: str
    fields: t.Mapping[str, t.Union[str, Encoder]]

    @cached_property
    def trivial(self):
        return any([isinstance(v, Encoder) for v in self.fields.values()])

    # TO DO - move this logic to ibis
    def on_create(self, db):
        if self.trivial:
            return
        assert not any(['_encodable' in k for k in self.fields.keys()]), 'Reserved substring: _encodable'
        schema_for_backend = {
            f'{k}::_encodable={v.identifier}/{v.version}::' if isinstance(v, Encoder) else k: v
            for k, v in self.fields.items() if isinstance(v, Encoder)
        }
        db.databackend.create_schema(self.identifier, schema_for_backend)

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
