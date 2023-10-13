import dataclasses as dc
import typing as t

from superduperdb.container.component import Component
from superduperdb.misc.serialization import serializers

if t.TYPE_CHECKING:
    from superduperdb.db.base.db import DB


@dc.dataclass
class Serializer(Component):
    identifier: str
    object: t.Type

    type_id: t.ClassVar[str] = 'serializer'

    version: t.Optional[int]

    def on_create(self, db: 'DB'):
        serializers.add(self.identifier, self.object)

        self.object = t.cast(t.Type, self.identifier)
